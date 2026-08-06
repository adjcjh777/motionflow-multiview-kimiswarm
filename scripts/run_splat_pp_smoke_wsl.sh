#!/usr/bin/env bash
# CPU smoke test for the Gaussian-splatting pose regularizer model.
# 2 epochs, batch size 2, small model. Logs to outputs/splat_pp_smoke.log.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE
# Force CPU even if CUDA drivers are present.
export CUDA_VISIBLE_DEVICES=""

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 \
    --d 16 \
    --residual_hidden 32 \
    --n_st_layers 2 \
    --model_type splat \
    --epochs 2 \
    --train_samples 50 \
    --batch_size 2 \
    --val_stride 20 \
    --pp_loss_weight 0.1 \
    --splat_loss_weight 0.05 \
    --cam_aug_pp 3.0 \
    --cam_aug_focal 0.01 \
    --output outputs/splat_pp_smoke.pth \
    > outputs/splat_pp_smoke.log 2>&1
