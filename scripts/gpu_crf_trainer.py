#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import numpy as np
except Exception as error:
    raise SystemExit(
        f"missing dependency `numpy`: {error}\n"
        "Install with: pip install numpy"
    )

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except Exception as error:
    raise SystemExit(
        f"missing dependency `torch`: {error}\n"
        "Install CUDA build with:\n"
        "  pip install --index-url https://download.pytorch.org/whl/cu124 torch"
    )


START_ROW = 0
END_ROW = 1
TRANSITION_OFFSET = 2


def parse_ini(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";") or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_mode_value(raw: str) -> Optional[str]:
    normalized = raw.strip().lower()
    if normalized in {"lac", "tag", "lexer"}:
        return "lac"
    if normalized in {"seg", "cutter"}:
        return "seg"
    return None


def load_word2id(path: Path) -> Dict[str, int]:
    values: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if "\t" not in line:
                continue
            raw_id, word = line.split("\t", 1)
            values[word] = int(raw_id)
    return values


def load_q2b(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if "\t" not in line:
                continue
            source, target = line.split("\t", 1)
            values[source] = target
    return values


def load_label_maps(path: Path) -> Tuple[Dict[str, int], List[str]]:
    label2id: Dict[str, int] = {}
    id2label_raw: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if "\t" not in line:
                continue
            raw_id, label = line.split("\t", 1)
            label_id = int(raw_id)
            label2id[label] = label_id
            id2label_raw[label_id] = label
    if not id2label_raw:
        raise ValueError(f"empty tag dictionary: {path}")
    num_labels = max(id2label_raw.keys()) + 1
    id2label = ["O"] * num_labels
    for label_id, label in id2label_raw.items():
        if 0 <= label_id < num_labels:
            id2label[label_id] = label
    return label2id, id2label


def infer_mode(parsed_args: Dict[str, str], id2label: List[str]) -> str:
    model_mode = parsed_args.get("model")
    if model_mode:
        parsed = parse_mode_value(model_mode)
        if parsed:
            return parsed
    tag_mode = parsed_args.get("tag_type")
    if tag_mode:
        parsed = parse_mode_value(tag_mode)
        if parsed:
            return parsed
    seg_labels = {"-B", "-I", "-E", "-S"}
    if id2label and all(label in seg_labels for label in id2label):
        return "seg"
    return "lac"


def read_paddle_tensor(path: Path) -> Tuple[np.ndarray, bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"invalid paddle tensor file (too small): {path}")
    header_len = int.from_bytes(raw[16:20], "little")
    payload_start = 20 + header_len
    if payload_start > len(raw):
        raise ValueError(f"invalid paddle tensor header length in: {path}")
    payload = raw[payload_start:]
    if len(payload) % 4 != 0:
        raise ValueError(f"invalid paddle tensor payload alignment in: {path}")
    values = np.frombuffer(payload, dtype="<f4").copy()
    prefix = raw[:payload_start]
    return values, prefix


@dataclass
class GruWeights:
    w_u: torch.Tensor
    w_r: torch.Tensor
    w_c: torch.Tensor
    b_u: torch.Tensor
    b_r: torch.Tensor
    b_c: torch.Tensor


@dataclass
class BigruLayer:
    fc_forward_w: torch.Tensor
    fc_forward_b: torch.Tensor
    gru_forward: GruWeights
    fc_reverse_w: torch.Tensor
    fc_reverse_b: torch.Tensor
    gru_reverse: GruWeights


class FrozenEmissionRuntime:
    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = model_dir
        self.device = device

        conf_dir = model_dir / "conf"
        param_dir = model_dir / "model"

        parsed_args = parse_ini(conf_dir / "args.ini")
        self.word_emb_dim = int(parsed_args["word_emb_dim"])
        self.hidden_dim = int(parsed_args["grnn_hidden_dim"])
        self.bigru_num = int(parsed_args["bigru_num"])

        self.word2id = load_word2id(conf_dir / "word.dic")
        self.q2b = load_q2b(conf_dir / "q2b.dic")
        self.label2id, self.id2label = load_label_maps(conf_dir / "tag.dic")
        self.mode = infer_mode(parsed_args, self.id2label)

        self.vocab_size = max(self.word2id.values()) + 1
        self.num_labels = len(self.id2label)
        self.oov_id = self.word2id.get("OOV", len(self.word2id) - 1)

        word_emb, _ = read_paddle_tensor(param_dir / "word_emb")
        self.word_emb = self._to_tensor(word_emb.reshape(self.vocab_size, self.word_emb_dim))

        self.layers: List[BigruLayer] = []
        for layer_id in range(self.bigru_num):
            input_dim = self.word_emb_dim if layer_id == 0 else self.hidden_dim * 2
            forward_idx = layer_id * 2
            reverse_idx = forward_idx + 1

            fc_forward_w, _ = read_paddle_tensor(param_dir / f"fc_{forward_idx}.w_0")
            fc_forward_b, _ = read_paddle_tensor(param_dir / f"fc_{forward_idx}.b_0")
            gru_forward_w, _ = read_paddle_tensor(param_dir / f"gru_{forward_idx}.w_0")
            gru_forward_b, _ = read_paddle_tensor(param_dir / f"gru_{forward_idx}.b_0")

            fc_reverse_w, _ = read_paddle_tensor(param_dir / f"fc_{reverse_idx}.w_0")
            fc_reverse_b, _ = read_paddle_tensor(param_dir / f"fc_{reverse_idx}.b_0")
            gru_reverse_w, _ = read_paddle_tensor(param_dir / f"gru_{reverse_idx}.w_0")
            gru_reverse_b, _ = read_paddle_tensor(param_dir / f"gru_{reverse_idx}.b_0")

            self.layers.append(
                BigruLayer(
                    fc_forward_w=self._to_tensor(fc_forward_w.reshape(input_dim, self.hidden_dim * 3)),
                    fc_forward_b=self._to_tensor(fc_forward_b.reshape(self.hidden_dim * 3)),
                    gru_forward=self._split_gru(gru_forward_w, gru_forward_b),
                    fc_reverse_w=self._to_tensor(fc_reverse_w.reshape(input_dim, self.hidden_dim * 3)),
                    fc_reverse_b=self._to_tensor(fc_reverse_b.reshape(self.hidden_dim * 3)),
                    gru_reverse=self._split_gru(gru_reverse_w, gru_reverse_b),
                )
            )

        emission_index = self.bigru_num * 2
        emission_w, _ = read_paddle_tensor(param_dir / f"fc_{emission_index}.w_0")
        emission_b, _ = read_paddle_tensor(param_dir / f"fc_{emission_index}.b_0")
        self.emission_w = self._to_tensor(emission_w.reshape(self.hidden_dim * 2, self.num_labels))
        self.emission_b = self._to_tensor(emission_b.reshape(self.num_labels))

        crf_raw, crf_prefix = read_paddle_tensor(param_dir / "crfw")
        expected_crf = (self.num_labels + 2) * self.num_labels
        if crf_raw.size != expected_crf:
            raise ValueError(
                f"invalid crfw shape: expected {expected_crf} values, got {crf_raw.size}"
            )
        self.initial_crf = crf_raw.reshape(self.num_labels + 2, self.num_labels)
        self.crf_prefix = crf_prefix

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.astype(np.float32, copy=False)).to(self.device)

    def _split_gru(self, weight_raw: np.ndarray, bias_raw: np.ndarray) -> GruWeights:
        hidden = self.hidden_dim
        gate_rows = hidden * hidden * 2
        gate_block = weight_raw[:gate_rows].reshape(hidden, hidden * 2)
        candidate = weight_raw[gate_rows:].reshape(hidden, hidden)
        bias = bias_raw.reshape(hidden * 3)
        return GruWeights(
            w_u=self._to_tensor(gate_block[:, :hidden]),
            w_r=self._to_tensor(gate_block[:, hidden:]),
            w_c=self._to_tensor(candidate),
            b_u=self._to_tensor(bias[:hidden]),
            b_r=self._to_tensor(bias[hidden : hidden * 2]),
            b_c=self._to_tensor(bias[hidden * 2 :]),
        )

    def clamp_token_id(self, value: int) -> int:
        if value < 0:
            return 0
        if value >= self.vocab_size:
            return self.vocab_size - 1
        return value

    def token_id_for_char(self, char: str) -> int:
        normalized = self.q2b.get(char, char)
        return self.clamp_token_id(self.word2id.get(normalized, self.oov_id))

    def fallback_label_id(self) -> int:
        if "O" in self.label2id:
            return min(self.label2id["O"], self.num_labels - 1)
        if "-S" in self.label2id:
            return min(self.label2id["-S"], self.num_labels - 1)
        return 0

    def label_id_or_fallback(self, label: str, fallback: int) -> int:
        return min(self.label2id.get(label, fallback), self.num_labels - 1)

    def parse_training_line(self, line: str) -> Optional[Tuple[List[int], List[int]]]:
        line = line.strip()
        if not line:
            return None
        if self.mode == "seg":
            return self._parse_seg_line(line)
        return self._parse_lac_line(line)

    def _parse_lac_line(self, line: str) -> Optional[Tuple[List[int], List[int]]]:
        token_ids: List[int] = []
        gold_ids: List[int] = []
        fallback = self.fallback_label_id()

        for item in line.split():
            if "/" not in item:
                return None
            word, tag = item.rsplit("/", 1)
            if not word or not tag:
                return None
            begin_id = self.label_id_or_fallback(f"{tag}-B", fallback)
            inside_id = self.label_id_or_fallback(f"{tag}-I", begin_id)
            for index, char in enumerate(word):
                token_ids.append(self.token_id_for_char(char))
                gold_ids.append(begin_id if index == 0 else inside_id)

        if not token_ids or len(token_ids) != len(gold_ids):
            return None
        return token_ids, gold_ids

    def _parse_seg_line(self, line: str) -> Optional[Tuple[List[int], List[int]]]:
        token_ids: List[int] = []
        gold_ids: List[int] = []
        fallback = self.fallback_label_id()
        begin_id = self.label_id_or_fallback("-B", fallback)
        inside_id = self.label_id_or_fallback("-I", begin_id)
        end_id = self.label_id_or_fallback("-E", inside_id)
        single_id = self.label_id_or_fallback("-S", end_id)

        for item in line.split():
            word = item.rsplit("/", 1)[0] if "/" in item else item
            if not word:
                continue
            chars = list(word)
            if len(chars) == 1:
                token_ids.append(self.token_id_for_char(chars[0]))
                gold_ids.append(single_id)
            elif len(chars) == 2:
                token_ids.append(self.token_id_for_char(chars[0]))
                gold_ids.append(begin_id)
                token_ids.append(self.token_id_for_char(chars[1]))
                gold_ids.append(end_id)
            else:
                for index, char in enumerate(chars):
                    token_ids.append(self.token_id_for_char(char))
                    if index == 0:
                        gold_ids.append(begin_id)
                    elif index == len(chars) - 1:
                        gold_ids.append(end_id)
                    else:
                        gold_ids.append(inside_id)

        if not token_ids or len(token_ids) != len(gold_ids):
            return None
        return token_ids, gold_ids

    def iter_rank_samples_once(
        self,
        data_path: Path,
        rank: int,
        world_size: int,
        max_samples: Optional[int],
    ) -> Iterator[Tuple[List[int], List[int]]]:
        usable_index = 0
        with data_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                sample = self.parse_training_line(raw_line)
                if sample is None:
                    continue
                if max_samples is not None and usable_index >= max_samples:
                    break
                if usable_index % world_size == rank:
                    yield sample
                usable_index += 1

    def iter_rank_samples_cycle(
        self,
        data_path: Path,
        rank: int,
        world_size: int,
        max_samples: Optional[int],
    ) -> Iterator[Tuple[List[int], List[int]]]:
        yielded_any = False
        while True:
            yielded_in_pass = False
            for sample in self.iter_rank_samples_once(data_path, rank, world_size, max_samples):
                yielded_any = True
                yielded_in_pass = True
                yield sample
            if not yielded_any:
                raise RuntimeError(f"no usable samples in {data_path}")
            if not yielded_in_pass:
                raise RuntimeError(
                    f"rank {rank} received no samples for {data_path}, reduce world size"
                )

    def count_usable_samples(self, data_path: Path, max_samples: Optional[int]) -> int:
        count = 0
        with data_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if self.parse_training_line(raw_line) is None:
                    continue
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        return count

    def dynamic_gru(self, pre: torch.Tensor, gru: GruWeights, reverse: bool) -> torch.Tensor:
        timesteps = pre.shape[0]
        hidden = self.hidden_dim
        pre_u, pre_r, pre_c = torch.split(pre, hidden, dim=1)
        output = torch.empty((timesteps, hidden), device=self.device, dtype=torch.float32)
        previous = torch.zeros(hidden, device=self.device, dtype=torch.float32)
        indices: Iterable[int] = range(timesteps - 1, -1, -1) if reverse else range(timesteps)

        for timestep in indices:
            gate_u = pre_u[timestep] + gru.b_u + previous @ gru.w_u
            gate_r = pre_r[timestep] + gru.b_r + previous @ gru.w_r
            update = torch.sigmoid(gate_u)
            reset = torch.sigmoid(gate_r)
            candidate = pre_c[timestep] + gru.b_c + (reset * previous) @ gru.w_c
            candidate = torch.tanh(candidate)
            current = update * candidate + (1.0 - update) * previous
            output[timestep] = current
            previous = current

        return output

    def compute_emission(self, token_ids: torch.Tensor) -> torch.Tensor:
        features = self.word_emb[token_ids]
        for layer in self.layers:
            pre_forward = features @ layer.fc_forward_w + layer.fc_forward_b
            pre_reverse = features @ layer.fc_reverse_w + layer.fc_reverse_b
            forward_out = self.dynamic_gru(pre_forward, layer.gru_forward, reverse=False)
            reverse_out = self.dynamic_gru(pre_reverse, layer.gru_reverse, reverse=True)
            features = torch.cat([forward_out, reverse_out], dim=1)
        return features @ self.emission_w + self.emission_b

    def save_crf_weights(self, transitions: np.ndarray) -> None:
        model_path = self.model_dir / "model" / "crfw"
        payload = transitions.astype("<f4", copy=False).reshape(-1).tobytes()
        temporary = model_path.with_suffix(".tmp")
        temporary.write_bytes(self.crf_prefix + payload)
        temporary.replace(model_path)


class CrfHead(torch.nn.Module):
    def __init__(self, initial_crf: np.ndarray, device: torch.device):
        super().__init__()
        initial = torch.from_numpy(initial_crf.astype(np.float32, copy=False)).to(device)
        self.transitions = torch.nn.Parameter(initial)
        self.num_labels = initial.shape[1]

    def sequence_score(self, emission: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        score = self.transitions[START_ROW, tags[0]] + emission[0, tags[0]]
        if tags.numel() > 1:
            previous = tags[:-1]
            current = tags[1:]
            score = score + self.transitions[TRANSITION_OFFSET + previous, current].sum()
            score = score + emission[1:, current].sum()
        score = score + self.transitions[END_ROW, tags[-1]]
        return score

    def log_partition(self, emission: torch.Tensor) -> torch.Tensor:
        alpha = self.transitions[START_ROW] + emission[0]
        transitions = self.transitions[TRANSITION_OFFSET:]
        for timestep in range(1, emission.shape[0]):
            scores = alpha.unsqueeze(1) + transitions
            alpha = torch.logsumexp(scores, dim=0) + emission[timestep]
        return torch.logsumexp(alpha + self.transitions[END_ROW], dim=0)

    def neg_log_likelihood(self, emission: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        return -(self.sequence_score(emission, tags) - self.log_partition(emission))

    @torch.no_grad()
    def viterbi_decode(self, emission: torch.Tensor) -> List[int]:
        timesteps = emission.shape[0]
        transitions = self.transitions[TRANSITION_OFFSET:]
        scores = self.transitions[START_ROW] + emission[0]
        backpointers: List[torch.Tensor] = []

        for timestep in range(1, timesteps):
            pair_scores = scores.unsqueeze(1) + transitions
            best_scores, best_previous = torch.max(pair_scores, dim=0)
            scores = best_scores + emission[timestep]
            backpointers.append(best_previous)

        scores = scores + self.transitions[END_ROW]
        best_last = int(torch.argmax(scores).item())
        path = [best_last]
        for previous in reversed(backpointers):
            best_last = int(previous[best_last].item())
            path.append(best_last)
        path.reverse()
        return path


@dataclass
class TrainMetrics:
    epochs: int
    samples: int
    tokens: int
    token_accuracy: float
    average_log_likelihood: float


def evaluate_dataset(
    runtime: FrozenEmissionRuntime,
    crf: CrfHead,
    eval_path: Path,
    max_eval_samples: Optional[int],
) -> TrainMetrics:
    total_samples = 0
    total_tokens = 0
    total_correct = 0
    total_log_likelihood = 0.0

    for token_ids, gold_ids in runtime.iter_rank_samples_once(eval_path, 0, 1, max_eval_samples):
        token_tensor = torch.tensor(token_ids, device=runtime.device, dtype=torch.long)
        gold_tensor = torch.tensor(gold_ids, device=runtime.device, dtype=torch.long)
        with torch.no_grad():
            emission = runtime.compute_emission(token_tensor)
            loss = crf.neg_log_likelihood(emission, gold_tensor)
            prediction = crf.viterbi_decode(emission)
        total_samples += 1
        total_tokens += len(gold_ids)
        total_correct += sum(1 for left, right in zip(prediction, gold_ids) if left == right)
        total_log_likelihood += float(-loss.item())

    if total_samples == 0:
        raise RuntimeError(f"no usable evaluation samples in {eval_path}")

    return TrainMetrics(
        epochs=1,
        samples=total_samples,
        tokens=total_tokens,
        token_accuracy=(total_correct / total_tokens) if total_tokens else 0.0,
        average_log_likelihood=total_log_likelihood / total_samples,
    )


def format_metrics(prefix: str, metrics: TrainMetrics) -> str:
    return (
        f"{prefix} samples={metrics.samples} tokens={metrics.tokens} "
        f"acc={metrics.token_accuracy:.4f} avg_ll={metrics.average_log_likelihood:.4f}"
    )


def worker_process(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    report_path: str,
) -> None:
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", args.master_addr)
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    use_cuda = args.gpus > 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    runtime = FrozenEmissionRuntime(Path(args.model_dir), device)
    crf = CrfHead(runtime.initial_crf, device)
    optimizer = torch.optim.AdamW([crf.transitions], lr=args.learning_rate)

    if rank == 0:
        print(
            f"[gpu-trainer] start mode={runtime.mode} gpus={world_size} epochs={args.epochs} "
            f"lr={args.learning_rate} train={args.train_data}"
        )

    if rank == 0:
        total_samples = runtime.count_usable_samples(Path(args.train_data), args.max_samples)
    else:
        total_samples = 0

    if world_size > 1:
        sample_tensor = torch.tensor([total_samples], device=device, dtype=torch.long)
        dist.broadcast(sample_tensor, src=0)
        total_samples = int(sample_tensor.item())

    if total_samples == 0:
        raise RuntimeError(f"no usable training samples in {args.train_data}")

    steps_per_epoch = max(1, math.ceil(total_samples / world_size))
    progress_interval = max(1, steps_per_epoch // 20)
    sample_iter = runtime.iter_rank_samples_cycle(
        Path(args.train_data), rank, world_size, args.max_samples
    )

    final_metrics = TrainMetrics(
        epochs=args.epochs,
        samples=0,
        tokens=0,
        token_accuracy=0.0,
        average_log_likelihood=0.0,
    )

    for epoch_index in range(args.epochs):
        local_samples = 0
        local_tokens = 0
        local_correct = 0
        local_log_likelihood = 0.0

        for step in range(steps_per_epoch):
            token_ids, gold_ids = next(sample_iter)
            token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
            gold_tensor = torch.tensor(gold_ids, device=device, dtype=torch.long)

            with torch.no_grad():
                emission = runtime.compute_emission(token_tensor)

            loss = crf.neg_log_likelihood(emission, gold_tensor)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if world_size > 1:
                dist.all_reduce(crf.transitions.grad, op=dist.ReduceOp.SUM)
                crf.transitions.grad /= float(world_size)
            optimizer.step()

            with torch.no_grad():
                prediction = crf.viterbi_decode(emission)

            local_samples += 1
            local_tokens += len(gold_ids)
            local_correct += sum(1 for left, right in zip(prediction, gold_ids) if left == right)
            local_log_likelihood += float(-loss.item())

            if rank == 0 and ((step + 1) % progress_interval == 0 or step + 1 == steps_per_epoch):
                progress = 100.0 * (step + 1) / steps_per_epoch
                accuracy = (local_correct / local_tokens) if local_tokens else 0.0
                average_ll = (local_log_likelihood / local_samples) if local_samples else 0.0
                print(
                    f"[train] epoch {epoch_index + 1}/{args.epochs} progress {progress:>5.1f}% "
                    f"({step + 1}/{steps_per_epoch}) acc={accuracy:.4f} avg_ll={average_ll:.4f}"
                )

        aggregate = torch.tensor(
            [local_samples, local_tokens, local_correct, local_log_likelihood],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)

        if rank == 0:
            epoch_samples = int(aggregate[0].item())
            epoch_tokens = int(aggregate[1].item())
            epoch_correct = int(aggregate[2].item())
            epoch_log_likelihood = float(aggregate[3].item())
            accuracy = (epoch_correct / epoch_tokens) if epoch_tokens else 0.0
            average_ll = (epoch_log_likelihood / epoch_samples) if epoch_samples else 0.0
            print(
                f"[train] epoch {epoch_index + 1}/{args.epochs} done "
                f"samples={epoch_samples} tokens={epoch_tokens} "
                f"acc={accuracy:.4f} avg_ll={average_ll:.4f}"
            )
            final_metrics = TrainMetrics(
                epochs=args.epochs,
                samples=epoch_samples * (epoch_index + 1),
                tokens=epoch_tokens * (epoch_index + 1),
                token_accuracy=accuracy,
                average_log_likelihood=average_ll,
            )

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        updated = crf.transitions.detach().to("cpu").numpy()
        runtime.save_crf_weights(updated)
        print(f"[gpu-trainer] saved updated crfw to {Path(args.model_dir) / 'model' / 'crfw'}")

        eval_metrics = evaluate_dataset(
            runtime=runtime,
            crf=crf,
            eval_path=Path(args.eval_data),
            max_eval_samples=args.max_eval_samples,
        )

        report = {
            "train": {
                "epochs": final_metrics.epochs,
                "samples": final_metrics.samples,
                "tokens": final_metrics.tokens,
                "token_accuracy": final_metrics.token_accuracy,
                "average_log_likelihood": final_metrics.average_log_likelihood,
            },
            "eval": {
                "epochs": eval_metrics.epochs,
                "samples": eval_metrics.samples,
                "tokens": eval_metrics.tokens,
                "token_accuracy": eval_metrics.token_accuracy,
                "average_log_likelihood": eval_metrics.average_log_likelihood,
            },
        }
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA CRF trainer for LAC model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("`--epochs` must be > 0")
    if args.learning_rate <= 0:
        raise SystemExit("`--learning-rate` must be > 0")
    if args.gpus < 0:
        raise SystemExit("`--gpus` must be >= 0")

    if args.gpus > 0 and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Install CUDA-enabled torch or set --gpus 0.")

    if args.gpus > torch.cuda.device_count():
        raise SystemExit(
            f"requested --gpus {args.gpus} but only {torch.cuda.device_count()} device(s) available"
        )

    world_size = args.gpus if args.gpus > 0 else 1
    report_file = tempfile.NamedTemporaryFile(
        prefix="lac_gpu_train_report_", suffix=".json", delete=False
    )
    report_file.close()

    if world_size > 1:
        mp.spawn(
            worker_process,
            nprocs=world_size,
            args=(world_size, args, report_file.name),
            join=True,
        )
    else:
        worker_process(0, world_size, args, report_file.name)

    report_path = Path(report_file.name)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.unlink(missing_ok=True)

    train_metrics = TrainMetrics(**report["train"])
    eval_metrics = TrainMetrics(**report["eval"])
    print(format_metrics("train", train_metrics))
    print(format_metrics("eval", eval_metrics))


if __name__ == "__main__":
    main()
