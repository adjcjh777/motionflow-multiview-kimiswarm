#!/usr/bin/env bash
# CPU smoke test for the view-dependent Gaussian-splatting pose regularizer.
# 2 epochs, batch size 2, small model. Logs to outputs/splatv2_pp_smoke.log.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=""

python -u experiments/train_splatv2_pp_smoke.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 \
    --d 16 \
    --residual_hidden 32 \
    --n_st_layers 2 \
    --epochs 2 \
    --train_samples 50 \
    --batch_size 2 \
    --val_stride 20 \
    --splat_loss_weight 0.05 \
    --output outputs/splatv2_pp_smoke.pth \
    > outputs/splatv2_pp_smoke.log 2>&1
