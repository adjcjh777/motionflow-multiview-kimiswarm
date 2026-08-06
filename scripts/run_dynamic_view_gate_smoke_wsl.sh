#!/usr/bin/env bash
# Smoke test for the dynamic view-selection gate (Tier-1/2 iter14 proposal).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 9 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --batch_size 8 --train_samples 500 --epochs 5 \
  --model_type dynamic_gate \
  --pp_loss_weight 0.1 \
  --gate_sparsity_weight 0.01 --gate_entropy_weight 0.001 \
  --view_noise_std 2.0 --joint_dropout_rate 0.15 \
  --output outputs/dynamic_view_gate_smoke.pth
