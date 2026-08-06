#!/usr/bin/env bash
# CPU smoke test for RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
# Force CPU-only execution for this smoke run.
export CUDA_VISIBLE_DEVICES=-1

mkdir -p outputs

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
  --clip_len 13 --d 32 --model_type camera_conditioned_pp --n_st_layers 2 \
  --residual_hidden 64 --epochs 2 --train_samples 100 --batch_size 2 --val_stride 20 \
  --pp_loss_weight 0.1 --cam_aug_pp 3.0 --cam_aug_focal 0.01 \
  --output outputs/camera_conditioned_pp_smoke_mpiinf3dhp.pth \
  "$@" \
  > outputs/camera_conditioned_smoke.log 2>&1
