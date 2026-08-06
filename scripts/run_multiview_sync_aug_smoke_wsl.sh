#!/usr/bin/env bash
# CPU smoke test for the view-synced temporal jitter augmentation wrapper.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
# Work around duplicate OpenMP runtime on the WSL / Git Bash runner.
export KMP_DUPLICATE_LIB_OK=TRUE

python -u experiments/train_multiview_sync_aug_smoke.py \
  --d 16 --residual_hidden 32 --n_st_layers 1 \
  --clip_len 5 --batch_size 2 --train_samples 20 --epochs 2 \
  --use_aug
