#!/usr/bin/env bash
# CPU smoke test for the deeper residual-gated temporal attention model.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

# Force CPU-only execution so the RTX 4090 remains free for full runs.
export CUDA_VISIBLE_DEVICES=""

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --n_view_layers 1 --n_temporal_layers 4 --n_view_groups 2 --n_joint_graph_layers 1 \
  --batch_size 2 --train_samples 200 --epochs 2 --val_stride 10 \
  --model_type deeper_temporal_pp \
  --pp_loss_weight 0.1 --cam_aug_pp 3.0 --cam_aug_focal 0.01 \
  --output outputs/deeper_temporal_pp_smoke.pth \
  > outputs/deeper_temporal_pp_smoke.log 2>&1
