#!/usr/bin/env bash
set -euo pipefail

# CPU smoke test for RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2.
export CUDA_VISIBLE_DEVICES=-1

LOG="outputs/epipolar_bias_v2_smoke.log"
mkdir -p "$(dirname "$LOG")"

python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train tmp/mpi_s01_seq01_smoke.npz \
    --val tmp/mpi_s02_seq01_smoke.npz \
    --clip_len 5 \
    --d 32 \
    --residual_hidden 64 \
    --n_st_layers 2 \
    --epochs 2 \
    --batch_size 2 \
    --train_samples 32 \
    --val_stride 1 \
    --pp_loss_weight 0.1 \
    --cam_aug_pp 2.0 \
    --cam_aug_focal 0.01 \
    --model_type epipolar_bias_v2_pp \
    --output outputs/epipolar_bias_v2_smoke.pth \
    --seed 42 \
    2>&1 | tee "$LOG"
