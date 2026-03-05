#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/gpu_crf_trainer.py" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../scripts/gpu_crf_trainer.py" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/../standalone/gpu_crf_trainer.py" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../standalone" && pwd)"
else
  echo "Không tìm thấy 'gpu_crf_trainer.py'." >&2
  echo "Hãy chạy: bash scripts/build-standalone-linux.sh" >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN_DEFAULT="${PYTHON_BIN}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN_DEFAULT="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "/opt/conda/bin/python" ]]; then
  PYTHON_BIN_DEFAULT="/opt/conda/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN_DEFAULT="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN_DEFAULT="$(command -v python)"
else
  echo "Không tìm thấy Python." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
TRAINER_PY="${TRAINER_PY:-${ROOT_DIR}/gpu_crf_trainer.py}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models_general/lac_model}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export GPU_MODEL="${GPU_MODEL:-RTX 4070}"
export GPU_COUNT="${GPU_COUNT:-2}"

LEARNING_RATE="${LEARNING_RATE:-0.01}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

if [[ ! -f "${TRAINER_PY}" ]]; then
  echo "Không tìm thấy trainer: ${TRAINER_PY}" >&2
  exit 1
fi

echo "[train-sequence] GPU setting: ${GPU_COUNT}x ${GPU_MODEL}"
echo "[train-sequence] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "[train-sequence] warning: không tìm thấy nvidia-smi."
fi
echo "[train-sequence] python=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import os
import sys
try:
    import torch
except Exception as error:
    sys.stderr.write(f"[train-sequence] torch import failed: {error}\n")
    sys.stderr.write("[train-sequence] install env: bash setup-gpu-env.sh\n")
    raise SystemExit(1)
if not torch.cuda.is_available():
    sys.stderr.write("[train-sequence] CUDA is not available in torch.\n")
    raise SystemExit(1)
visible = torch.cuda.device_count()
sys.stderr.write(f"[train-sequence] torch={torch.__version__} cuda={torch.version.cuda}\n")
sys.stderr.write(f"[train-sequence] visible_gpu_count={visible}\n")
requested = int(os.environ.get("GPU_COUNT", "1"))
if visible < requested:
    sys.stderr.write(
        f"[train-sequence] requested GPU_COUNT={requested} but only {visible} visible GPU(s).\n"
    )
    raise SystemExit(1)
for index in range(visible):
    sys.stderr.write(f"[train-sequence] gpu[{index}]={torch.cuda.get_device_name(index)}\n")
PY

run_stage() {
  local dataset_name="$1"
  local train_file="$2"
  local eval_file="$3"
  local epochs="$4"

  echo
  echo "[train-sequence] start ${dataset_name} (epochs=${epochs}, lr=${LEARNING_RATE})"
  local cmd=(
    "${PYTHON_BIN}" "${TRAINER_PY}"
    --model-dir "${MODEL_DIR}"
    --train-data "${DATA_DIR}/${train_file}"
    --eval-data "${DATA_DIR}/${eval_file}"
    --epochs "${epochs}"
    --learning-rate "${LEARNING_RATE}"
    --gpus "${GPU_COUNT}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}" --max-eval-samples "${MAX_SAMPLES}")
  fi
  "${cmd[@]}"
}

run_stage "texsmart" "lac_train_texsmart.tsv" "lac_test_texsmart.tsv" "10"
run_stage "zh_gsdsimp" "lac_train_zh_gsdsimp.tsv" "lac_test_zh_gsdsimp.tsv" "20"
run_stage "chivi" "lac_train_chivi.tsv" "lac_test_chivi.tsv" "20"

echo
echo "[train-sequence] done."
