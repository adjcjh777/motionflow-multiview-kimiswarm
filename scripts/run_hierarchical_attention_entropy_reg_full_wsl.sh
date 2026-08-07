#!/usr/bin/env bash
# Full GPU training for hierarchical attention + attention-entropy regularisation.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

mkdir -p outputs

python -u experiments/train_hierarchical_attention_entropy_reg_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --n_view_layers 2 --n_temporal_layers 2 --n_view_groups 2 --n_joint_graph_layers 1 \
  --batch_size 8 --train_samples 1000 --epochs 20 --val_stride 50 \
  --pp_loss_weight 0.2 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
  --pp_pretrain_epochs 3 --attention_entropy_weight 0.01 \
  --output outputs/hierarchical_attention_entropy_reg_full_mpiinf3dhp.pth
