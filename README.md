# lac-rs

Rust-native port of Baidu LAC (`lac-master`) using `models_general` weights.

## What is included

- Native Rust inference for:
  - `lac_model`: segmentation + POS/NER tags
  - `seg_model`: segmentation only
- No Paddle / Python runtime dependency at inference time.
- `wasm-bindgen` API for browser/Node targets.
- `models_general` is embedded with `include_bytes!`, so wasm builds carry the model inside `.wasm`.

## Quick start (Rust)

```rust
use lac_rs::{Lac, Mode};

let lac = Lac::new(Mode::Lac)?;
let result = lac.run("百度是一家高科技公司");
```

## Build

```bash
cargo build
```

## Build Linux standalone package

```bash
bash scripts/build-standalone-linux.sh
```

Sau khi build xong:
- binary: `standalone/bin/lac-rs-train`
- model: `standalone/models_general/lac_model`
- data: `standalone/data/*.tsv`
- CUDA trainer: `standalone/gpu_crf_trainer.py`
- setup env script: `standalone/setup-gpu-env.sh`
- sequence train script: `standalone/train-sequence-gpu.sh` (texsmart 10 → zh_gsdsimp 20 → chivi 20)
- file upload VPS: `lac-rs-standalone-linux-<arch>.tar.gz`

Deploy nhanh lên VPS:
```bash
scp lac-rs-standalone-linux-<arch>.tar.gz user@vps:/path/
ssh user@vps
tar -xzf lac-rs-standalone-linux-<arch>.tar.gz
cd standalone
# VastAI image: vastai/pytorch_cuda-12.6.3-auto/jupyter
GPU_COUNT=2 GPU_MODEL="RTX 4070" bash train-sequence-gpu.sh

# Nếu thiếu torch/cuda trong python env thì setup fallback:
bash setup-gpu-env.sh
GPU_COUNT=2 GPU_MODEL="RTX 4070" bash train-sequence-gpu.sh
```

## Train with `data/` dataset (native Rust)

Training runs native in Rust and fine-tunes only `crfw` (CRF transition weights), while keeping BiGRU emissions frozen.

```bash
cargo run --example train
```

Default paths:
- model: `models_general/lac_model`
- train: `data/lac_train_chivi.tsv`
- eval: `data/lac_test_chivi.tsv`

Custom paths/params:

```bash
cargo run --example train -- <model_dir> <train_tsv> <eval_tsv> <epochs> <learning_rate> [max_samples]
```

## Build wasm (model embedded)

```bash
cargo build --release --target wasm32-unknown-unknown --features wasm
```

## Web packaging (wasm-pack)

```bash
cargo install wasm-pack
bash scripts/build-web.sh
```

Kết quả JS + wasm sẽ nằm ở `web/pkg/`.

## Chạy demo web

```bash
python3 -m http.server 8080
```

Mở `http://localhost:8080/web/` để test `textarea` demo (`web/index.html`).
