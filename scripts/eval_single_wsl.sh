#!/usr/bin/env bash
# Evaluate a single checkpoint on MPI-INF-3DHP validation.
# Usage: scripts/eval_single_wsl.sh <checkpoint> <model_type> <d> [residual_hidden] [output_json]
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

CHECKPOINT="$1"
MODEL="$2"
D="${3:-128}"
RES_HID="${4:-256}"
OUT_JSON="${5:-outputs/eval_$(basename "$CHECKPOINT" .pth).json}"

python -u experiments/eval_full_metrics.py \
  --model "$MODEL" \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --checkpoint "$CHECKPOINT" \
  --clip_len 13 --d "$D" --residual_hidden "$RES_HID" --n_st_layers 3 \
  --batch_size 4 --val_stride 50 \
  --output_json "$OUT_JSON"
