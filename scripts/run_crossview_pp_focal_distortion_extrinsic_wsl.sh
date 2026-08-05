#!/usr/bin/env bash
# GPU launcher skeleton: stronger calibration-robustness curriculum for the
# cross-view PP model.  Adds focal-length, radial-distortion and stronger
# extrinsic perturbations to the existing principal-point curriculum.
#
# DO NOT RUN NOW — the RTX 4090 is busy with the cross-view PP curriculum.
# When the GPU is free, start with a short smoke run, then the full run.
set -e
cd "$(dirname "$0")/.."

# Adjust to the local venv if needed.
VENV="${VENV:-/tmp/mf_venv/bin/activate}"
if [ -f "$VENV" ]; then
  # shellcheck source=/dev/null
  . "$VENV"
fi

SMOKE_SAMPLES=500
FULL_SAMPLES=4000
TRAIN_SAMPLES="${SMOKE_SAMPLES}"

# Change to --train_samples "$FULL_SAMPLES" for the real experiment.
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
    --epochs 20 --batch_size 8 --train_samples "$TRAIN_SAMPLES" --val_stride 50 \
    --pp_loss_weight 0.05 --focal_max_scale 0.02 --focal_loss_weight 0.05 \
    --cam_aug_rot_max 1.0 --cam_aug_trans_max 0.02 \
    --cam_aug_focal_max 0.02 --cam_aug_pp_max 5.0 \
    --cam_aug_k1_std 0.05 \
    --view_dropout_rate 0.2 --min_views 4 \
    --output "outputs/ray_attention_temporal_crossview_residual_pp_focal_distortion_extrinsic_v1.pth"
