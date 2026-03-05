#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "${SCRIPT_DIR}/.venv" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/../standalone" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../standalone" && pwd)"
else
  ROOT_DIR="${SCRIPT_DIR}"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN_DEFAULT="${PYTHON_BIN}"
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
VENV_DIR="${ROOT_DIR}/.venv"

if "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception:
    raise SystemExit(1)
if not torch.cuda.is_available():
    raise SystemExit(1)
print(f"[setup-gpu-env] reuse existing torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[setup-gpu-env] visible_gpu_count={torch.cuda.device_count()}")
PY
then
  echo "[setup-gpu-env] environment already ready, skip install."
  exit 0
fi

echo "[setup-gpu-env] current python chưa có torch cuda, tạo venv fallback..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch

echo "[setup-gpu-env] done (fallback venv)"
echo "[setup-gpu-env] activate with: source ${VENV_DIR}/bin/activate"
