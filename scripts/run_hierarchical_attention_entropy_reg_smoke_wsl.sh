#!/usr/bin/env bash
# CPU smoke test for the hierarchical attention + attention-entropy regularisation model.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

# Force CPU-only execution so the RTX 4090 remains free for the full run.
export CUDA_VISIBLE_DEVICES=""

python -u experiments/train_hierarchical_attention_entropy_reg_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --n_view_layers 1 --n_temporal_layers 1 --n_view_groups 2 --n_joint_graph_layers 1 \
  --batch_size 2 --train_samples 200 --epochs 2 --val_stride 10 \
  --attention_entropy_weight 0.01 \
  --pp_loss_weight 0.1 --cam_aug_pp 3.0 --cam_aug_focal 0.01 \
  --output outputs/hierarchical_attention_entropy_reg_smoke.pth \
  > outputs/hierarchical_attention_entropy_reg_smoke.log 2>&1

echo "Smoke test complete. Log: outputs/hierarchical_attention_entropy_reg_smoke.log"
