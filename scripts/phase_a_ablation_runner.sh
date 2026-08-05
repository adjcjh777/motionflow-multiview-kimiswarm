#!/usr/bin/env bash
# Phase A ablation: loss architecture / pp_loss_weight sweep.
# Runs sequentially on a single RTX 4090 to avoid GPU OOM.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

TRAIN=(
  data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz
)
VAL=data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz

COMMON_FLAGS=(
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64
  --epochs 10 --train_samples 500 --val_stride 50 --batch_size 8
  --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 5.0
)

# A1: 3D MSE only (no pp correction head effectively)
python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train "${TRAIN[@]}" --val "$VAL" "${COMMON_FLAGS[@]}" \
  --pp_loss_weight 0.0 --reproj_weight 0.0 \
  --output outputs/pp_ablation_A1_baseline.pth

# A2: reprojection consistency only
python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train "${TRAIN[@]}" --val "$VAL" "${COMMON_FLAGS[@]}" \
  --pp_loss_weight 0.0 --reproj_weight 1.0 \
  --output outputs/pp_ablation_A2_reproj_only.pth

# A3: balanced MSE+reproj
python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train "${TRAIN[@]}" --val "$VAL" "${COMMON_FLAGS[@]}" \
  --pp_loss_weight 0.0 --reproj_weight 0.1 \
  --output outputs/pp_ablation_A3_mse_reproj.pth

# A4: explicit offset (re-run for clean baseline)
python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train "${TRAIN[@]}" --val "$VAL" "${COMMON_FLAGS[@]}" \
  --pp_loss_weight 0.1 --reproj_weight 0.0 \
  --output outputs/pp_ablation_A4_explicit_0.1.pth

# A5: explicit + reprojection
python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train "${TRAIN[@]}" --val "$VAL" "${COMMON_FLAGS[@]}" \
  --pp_loss_weight 0.1 --reproj_weight 0.5 \
  --output outputs/pp_ablation_A5_explicit_reproj.pth

# A6: higher explicit weight
python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train "${TRAIN[@]}" --val "$VAL" "${COMMON_FLAGS[@]}" \
  --pp_loss_weight 0.5 --reproj_weight 0.0 \
  --output outputs/pp_ablation_A6_explicit_0.5.pth

echo "Phase A ablation training complete."
