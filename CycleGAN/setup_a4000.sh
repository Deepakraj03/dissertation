#!/usr/bin/env bash
# Portable environment setup for training CycleGAN on a CUDA GPU machine
# (written for an RTX A4000, but works on any CUDA-capable box).
#
# Usage (on the target machine, from the project root):
#   ./setup_a4000.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "=== CycleGAN GPU setup ==="

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install the NVIDIA driver first."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
echo "Detected driver CUDA version: ${CUDA_VER:-unknown}"

if [[ -z "$CUDA_VER" ]]; then
    WHEEL_TAG="cu121"
elif awk "BEGIN{exit !($CUDA_VER >= 12.4)}"; then
    WHEEL_TAG="cu124"
elif awk "BEGIN{exit !($CUDA_VER >= 12.1)}"; then
    WHEEL_TAG="cu121"
else
    WHEEL_TAG="cu118"
fi
echo "Selected PyTorch wheel index: $WHEEL_TAG"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo ""
echo "=== Installing PyTorch (CUDA build) ==="
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${WHEEL_TAG}"

echo ""
echo "=== Installing remaining dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Verifying CUDA is visible to PyTorch ==="
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print("WARNING: CUDA not available — check driver/wheel match (tried tag: "
          f"{__import__('os').environ.get('WHEEL_TAG', '?')})")
PY

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
