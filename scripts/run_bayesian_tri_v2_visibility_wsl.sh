#!/usr/bin/env bash
# Bayesian Tri v2 with learned per-view/per-joint visibility gating.
# Warm-starts from the best stabilized d=128 checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

mkdir -p outputs

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_04_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_04_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_05_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_05_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_06_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_06_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_07_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_07_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_08_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_08_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 128 --residual_hidden 256 --n_st_layers 3 \
  --model_type bayesian_tri_v2_visibility \
  --epochs 30 --train_samples 2000 --batch_size 6 --val_stride 50 \
  --lr 3e-4 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
  --grad_clip_norm 1.0 \
  --pp_loss_weight 0.2 --epipolar_loss_weight 0.05 --reproj_weight 0.0 \
  --visibility_loss_weight 0.1 \
  --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule extended_intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
  --view_dropout_rate 0.2 --min_views 4 \
  --pp_pretrain_epochs 0 \
  --warm_start outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth \
  --output outputs/bayesian_tri_v2_visibility_mpiinf3dhp.pth
