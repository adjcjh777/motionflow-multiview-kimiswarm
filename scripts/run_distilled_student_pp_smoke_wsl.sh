#!/usr/bin/env bash
# CPU smoke test for the knowledge-distilled lightweight student.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=""

python -u experiments/train_distilled_student_pp_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
  --teacher outputs/bayesian_tri_pp_full_mpiinf3dhp.pth \
  --clip_len 13 --d 32 --n_st_layers 1 --residual_hidden 64 \
  --distill_alpha 0.5 --weight_align_beta 0.1 \
  --batch_size 2 --train_samples 50 --epochs 1 --val_stride 20 \
  --pp_loss_weight 0.1 --cam_aug_pp 3.0 --cam_aug_focal 0.01 \
  --pp_pretrain_epochs 0 \
  --output outputs/distilled_student_pp_smoke.pth \
  > outputs/distilled_student_pp_smoke.log 2>&1

echo "Smoke test passed: outputs/distilled_student_pp_smoke.log"
