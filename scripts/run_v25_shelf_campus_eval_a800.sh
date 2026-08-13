#!/usr/bin/env bash
# Evaluate the v25 true-GT v2 medium checkpoint on Shelf/Campus detected val data (A800-D).
#
# The checkpoint is produced by scripts/run_v25_true_gt_v2_medium_a800.sh and is
# located at outputs/ablations/v25_true_gt_v2_medium_a800.pth.
#
# Usage
# -----
#   bash scripts/run_v25_shelf_campus_eval_a800.sh
#
#   # Run on a different GPU
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v25_shelf_campus_eval_a800.sh
set -euo pipefail

# Pin to the A800-D repo where the checkpoint lives.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

CKPT="outputs/ablations/v25_true_gt_v2_medium_a800.pth"
OUT_JSON="outputs/eval_v25_true_gt_v2_shelf_campus.json"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT" >&2
    exit 1
fi

echo "Evaluating v25 true-GT v2 checkpoint on Shelf/Campus val split (GPU ${CUDA_VISIBLE_DEVICES})"

$PYTHON -u experiments/eval_v25_cross_dataset.py \
    --checkpoint "$CKPT" \
    --split configs/splits/v25_shelf_campus_eval.yaml \
    --clip_len 13 \
    --batch_size 8 \
    --val_stride 13 \
    --out_json "$OUT_JSON"

echo "Done. Results: $OUT_JSON"
