#!/usr/bin/env bash
# Usage: bash setup/run_setup_test.sh [DATA_ROOT]
# Dataset layout check, then a ~10 min smoke test; exit code 0 means pass.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${1:-"${ROOT}/dataset"}

run() { conda run --no-capture-output -n anomaly python "$@"; }

echo "=== [1/2] Dataset check ==="
run "${ROOT}/scripts/check_dataset.py" --data-root "$DATA_ROOT" || exit 1

echo "=== [2/2] Smoke test ==="
run "${ROOT}/scripts/smoke_test.py" --data-root "$DATA_ROOT" || exit 1

echo
echo "Setup test PASSED. Launch the main experiment with:"
echo "  nohup python ${ROOT}/scripts/run_all.py > run.log 2>&1 &"
