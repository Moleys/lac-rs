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
