#!/usr/bin/env bash
# Usage: bash setup/setup_env.sh
set -euo pipefail

ENV_NAME=anomaly
PYTHON_VERSION=3.11
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Load it first (e.g. 'module load anaconda3'"
    echo "on most university clusters) and re-run this script."
    exit 1
fi

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "conda env '${ENV_NAME}' already exists -- reusing it."
else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

run() { conda run --no-capture-output -n "${ENV_NAME}" "$@"; }

echo ">>> Installing dependencies (including the CUDA build of torch)..."
run pip install -r "${ROOT}/requirements.txt"

echo ">>> Verifying installation..."
run python - <<'EOF'
import torch
from anomalib.models import EfficientAd, Ganomaly, Padim, Patchcore, ReverseDistillation

print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit(
        "ERROR: CUDA not available. A dependency likely replaced torch with a "
        "CPU build -- re-run pip install -r requirements.txt, or check "
        "that this node has a GPU and a recent NVIDIA driver."
    )
print("All five models import correctly.")
EOF

echo
echo "Setup complete. Next steps:"
echo "  1. Place MVTec AD under ./dataset (see README)."
echo "  2. bash setup/run_setup_test.sh"
