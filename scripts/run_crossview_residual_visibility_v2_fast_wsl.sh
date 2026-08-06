#!/usr/bin/env bash
# Visibility-gated cross-view residual v2 training on MPI-INF-3DHP (faster val).
set -e
cd "$(dirname "$0")/.."

. /tmp/mf_venv/bin/activate
export PYTHONUNBUFFERED=1

python -u experiments/train_crossview_residual_visibility_v2_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
    --epochs 20 --batch_size 8 --train_samples 4000 --val_stride 10 \
    --pp_loss_weight 0.05 --cam_aug_pp 5.0 \
    --view_dropout_rate 0.2 --min_views 4 \
    --visibility_loss_weight 0.1 \
    --output outputs/ray_attention_temporal_crossview_residual_visibility_v2_mpiinf3dhp.pth \
    > outputs/ray_attention_temporal_crossview_residual_visibility_v2_mpiinf3dhp.log 2>&1
