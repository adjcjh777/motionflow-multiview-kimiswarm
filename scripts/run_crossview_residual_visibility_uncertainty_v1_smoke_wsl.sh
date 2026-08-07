#!/usr/bin/env bash
# CPU/GPU smoke training for the visibility + uncertainty v1 model.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

LOG="outputs/crossview_residual_visibility_uncertainty_v1_smoke.log"
mkdir -p "$(dirname "$LOG")"

python -u experiments/train_crossview_residual_visibility_uncertainty_v1_mpiinf3dhp.py     --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz     --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz     --clip_len 13 --d 32 --n_st_layers 1 --residual_hidden 64     --epochs 2 --batch_size 4 --train_samples 256 --val_stride 10 --num_workers 0     --pp_loss_weight 0.05 --cam_aug_pp 2.0     --view_dropout_rate 0.2 --min_views 4     --visibility_loss_weight 0.1 --uncertainty_loss_weight 0.1     --output outputs/crossview_residual_visibility_uncertainty_v1_smoke.pth     > "$LOG" 2>&1
