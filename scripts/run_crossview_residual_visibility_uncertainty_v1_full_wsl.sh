#!/usr/bin/env bash
# Full 30-epoch training for the combined visibility + uncertainty cross-view residual v1 model.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

LOG="outputs/crossview_residual_visibility_uncertainty_v1_full_mpiinf3dhp.log"
mkdir -p "$(dirname "$LOG")"

python -u experiments/train_crossview_residual_visibility_uncertainty_v1_mpiinf3dhp.py     --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz            data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz            data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz     --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz     --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128     --epochs 30 --batch_size 8 --train_samples 4000 --val_stride 1     --pp_loss_weight 0.05 --cam_aug_pp 5.0     --view_dropout_rate 0.2 --min_views 4     --visibility_loss_weight 0.1 --uncertainty_loss_weight 0.1     --output outputs/crossview_residual_visibility_uncertainty_v1_mpiinf3dhp.pth     > "$LOG" 2>&1
