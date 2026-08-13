#!/usr/bin/env bash
# Run inference with the v25 true-GT v2 medium checkpoint on MPI-INF-3DHP detected-2D data (A800-D).
#
# The checkpoint is produced by scripts/run_v25_true_gt_v2_medium_a800.sh and is
# located at outputs/ablations/v25_true_gt_v2_medium_a800.pth.
#
# This is prediction-only: MPI-INF-3DHP official test-set GT is not available locally,
# so the script saves predictions to a .npz file and does not report MPJPE.
#
# Usage
# -----
#   bash scripts/run_v25_mpi_detected_eval_a800.sh
#
#   # Run on a different GPU
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v25_mpi_detected_eval_a800.sh
set -euo pipefail

# Pin to the A800-D repo where the checkpoint lives.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

CKPT="outputs/ablations/v25_true_gt_v2_medium_a800.pth"
OUT_JSON="outputs/eval_v25_true_gt_v2_mpi.json"
OUT_NPZ="outputs/eval_v25_true_gt_v2_mpi_predictions.npz"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT" >&2
    exit 1
fi

echo "Running v25 true-GT v2 inference on MPI-INF-3DHP detected-2D (GPU ${CUDA_VISIBLE_DEVICES})"

$PYTHON -u experiments/eval_v25_cross_dataset.py \
    --checkpoint "$CKPT" \
    --dataset_name mpi \
    --split configs/splits/v25_mpi_detected_eval.yaml \
    --clip_len 13 \
    --batch_size 8 \
    --val_stride 13 \
    --infer_only \
    --out_json "$OUT_JSON" \
    --out_npz "$OUT_NPZ"

echo "Done. Results: $OUT_JSON"
echo "Predictions: $OUT_NPZ"
