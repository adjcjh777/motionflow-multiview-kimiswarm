#!/usr/bin/env bash
# CPU smoke test for the semi-supervised teacher-student pseudo-label trainer.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

# Force CPU-only execution so the RTX 4090 remains free for full GPU runs.
export CUDA_VISIBLE_DEVICES=""

python -u tests/test_pseudo_label_training.py
