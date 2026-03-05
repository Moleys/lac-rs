#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDALONE_DIR="${ROOT_DIR}/standalone"
ARCH="$(uname -m)"
PACKAGE_NAME="lac-rs-standalone-linux-${ARCH}.tar.gz"
PACKAGE_PATH="${ROOT_DIR}/${PACKAGE_NAME}"

MODEL_SRC_DIR="${ROOT_DIR}/models_general/lac_model"
MODEL_DST_DIR="${STANDALONE_DIR}/models_general/lac_model"
DATA_SRC_DIR="${ROOT_DIR}/data"
DATA_DST_DIR="${STANDALONE_DIR}/data"
BIN_SRC="${ROOT_DIR}/target/release/examples/train"
BIN_DST="${STANDALONE_DIR}/bin/lac-rs-train"

DATA_FILES=(
  "lac_train_texsmart.tsv"
  "lac_test_texsmart.tsv"
  "lac_train_zh_gsdsimp.tsv"
  "lac_test_zh_gsdsimp.tsv"
  "lac_train_chivi.tsv"
  "lac_test_chivi.tsv"
)

if [[ ! -d "${MODEL_SRC_DIR}" ]]; then
  echo "Không tìm thấy model: ${MODEL_SRC_DIR}" >&2
  exit 1
fi

echo "[standalone] build linux release binary..."
cargo build --release --example train

echo "[standalone] prepare folder: ${STANDALONE_DIR}"
rm -rf "${STANDALONE_DIR}"
mkdir -p "${STANDALONE_DIR}/bin" "${STANDALONE_DIR}/models_general" "${DATA_DST_DIR}"

echo "[standalone] copy binary..."
cp "${BIN_SRC}" "${BIN_DST}"
chmod +x "${BIN_DST}"

echo "[standalone] copy model..."
cp -a "${MODEL_SRC_DIR}" "${MODEL_DST_DIR}"

echo "[standalone] copy datasets..."
for data_file in "${DATA_FILES[@]}"; do
  cp "${DATA_SRC_DIR}/${data_file}" "${DATA_DST_DIR}/${data_file}"
done

echo "[standalone] copy train scripts..."
cp "${ROOT_DIR}/scripts/train-sequence-gpu.sh" "${STANDALONE_DIR}/train-sequence-gpu.sh"
cp "${ROOT_DIR}/scripts/gpu_crf_trainer.py" "${STANDALONE_DIR}/gpu_crf_trainer.py"
cp "${ROOT_DIR}/scripts/setup-gpu-env.sh" "${STANDALONE_DIR}/setup-gpu-env.sh"
chmod +x "${STANDALONE_DIR}/train-sequence-gpu.sh"
chmod +x "${STANDALONE_DIR}/gpu_crf_trainer.py"
chmod +x "${STANDALONE_DIR}/setup-gpu-env.sh"

cat > "${STANDALONE_DIR}/README.md" <<'EOF'
# standalone

## Build package
```bash
bash scripts/build-standalone-linux.sh
```

## Run training sequence
```bash
cd standalone
# VastAI image (vastai/pytorch_cuda-12.6.3-auto/jupyter):
GPU_COUNT=2 GPU_MODEL="RTX 4070" bash train-sequence-gpu.sh

# Nếu thiếu torch/cuda trong python env:
bash setup-gpu-env.sh
GPU_COUNT=2 GPU_MODEL="RTX 4070" bash train-sequence-gpu.sh
```
EOF

echo "[standalone] create archive..."
rm -f "${PACKAGE_PATH}"
tar -C "${ROOT_DIR}" -czf "${PACKAGE_PATH}" standalone

echo "[standalone] done."
echo "  Binary : ${BIN_DST}"
echo "  Model  : ${MODEL_DST_DIR}"
echo "  Data   : ${DATA_DST_DIR}"
echo "  Archive: ${PACKAGE_PATH}"
