#!/usr/bin/env bash
# 10-epoch smoke train for the cross-view residual + principal-point + uncertainty model.
# This script is intentionally GPU-only and is queued behind the currently running
# cross-view PP curriculum.  Do NOT run until the RTX 4090 is free.
set -e
cd "$(dirname "$0")/.."

. /tmp/mf_venv/bin/activate

python experiments/train_crossview_residual_uncertainty_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
    --epochs 10 --batch_size 8 --train_samples 4000 --val_stride 1 \
    --uncertainty_loss_weight 0.1 \
    --pp_loss_weight 0.0 --focal_max_scale 0.0 \
    --cam_aug_pp 5.0 --cam_aug_schedule extrinsic_curriculum --cam_aug_ramp_epochs 5 \
    --view_dropout_rate 0.2 --min_views 4 \
    --output outputs/crossview_residual_uncertainty_smoke_v1.pth
