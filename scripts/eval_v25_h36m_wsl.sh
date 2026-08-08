#!/usr/bin/env bash
# Evaluate a v25 OmniMultiViewFusionV5 checkpoint on Human3.6M.
#
# The v25 architecture flags are read automatically from the side-car
# ``<checkpoint>.config.json`` produced during training, so only the
# checkpoint and dataset paths need to be supplied.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

CHECKPOINT=${CHECKPOINT:-outputs/omniview_fusion_v25_geometry_fusion_small.pth}
DATASET=${DATASET:-data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz}

python -u experiments/eval_omniview_fusion_v5_h36m.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --run_robustness \
  --run_variable_views \
  --clip_len 13 --batch_size 8 --val_stride 10 \
  --out_json outputs/eval_v25_h36m.json \
  --out_csv outputs/eval_v25_h36m.csv \
  "$@"
