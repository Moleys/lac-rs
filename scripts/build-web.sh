#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack chưa được cài. Cài bằng: cargo install wasm-pack"
  exit 1
fi

wasm-pack build \
  --release \
  --target web \
  --out-dir web/pkg \
  --out-name lac_rs \
  -- --features wasm

echo "Build xong: web/pkg/"
