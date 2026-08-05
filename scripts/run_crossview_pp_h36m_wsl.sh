#!/usr/bin/env bash
# Cross-view temporal residual + PP on Human3.6M (WSL + RTX 4090).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

# Train on subject 1 actions 03-16, validate on subject 1 action 02.
TRAIN=""
for a in 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  TRAIN="$TRAIN data/webbridge/h36m_meters/s_01_acts_${a}_multiview_m.npz"
done

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train $TRAIN \
  --val data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 --epochs 10 \
  --train_samples 500 --batch_size 8 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --output outputs/ray_attention_temporal_crossview_residual_principal_point_h36m_small.pth \
  "$@"
