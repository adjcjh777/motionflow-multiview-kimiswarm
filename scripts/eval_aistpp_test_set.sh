#!/usr/bin/env bash
# Evaluate a trained OmniMultiViewFusionV5 checkpoint on the AIST++ test split.
#
# Uses configs/splits/aistpp_train_val_test.yaml and the matching checkpoint.
# Runs on the A800-D repo; targets GPU 6 by default (override with
# CUDA_VISIBLE_DEVICES).  CPU-only inference is also supported.
#
# Usage
# -----
#   bash scripts/eval_aistpp_test_set.sh v25
#   bash scripts/eval_aistpp_test_set.sh v80
#   bash scripts/eval_aistpp_test_set.sh v57
#
#   # Override checkpoint / GPU
#   VARIANT=v80 CKPT=outputs/ablations/v80_aistpp_full_medium_a800.pth \
#       bash scripts/eval_aistpp_test_set.sh v80
set -euo pipefail

VARIANT=${1:-v25}
CKPT_BASE="outputs/ablations/${VARIANT}_aistpp_full_medium_a800"
# Prefer an explicit CKPT override, then the symlink/regular .pth, then the final checkpoint.
if [[ -n "${CKPT:-}" ]]; then
    :
elif [[ -f "${CKPT_BASE}.pth" ]]; then
    CKPT="${CKPT_BASE}.pth"
elif [[ -f "${CKPT_BASE}_final.pth" ]]; then
    CKPT="${CKPT_BASE}_final.pth"
else
    CKPT="${CKPT_BASE}.pth"
fi
SPLIT=${SPLIT:-configs/splits/aistpp_train_val_test.yaml}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT" >&2
    echo "       Train the model first, then re-run this script." >&2
    exit 1
fi

OUT_JSON="outputs/ablations/${VARIANT}_aistpp_full_medium_a800_test.json"

# Match the clip_len used during training.
case "$VARIANT" in
  v80) CLIP_LEN=9 ;;
  *)   CLIP_LEN=13 ;;
esac

$PYTHON -u experiments/eval_omniview_fusion_v5_aistpp.py \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --clip_len "$CLIP_LEN" \
    --batch_size 8 \
    --out_json "$OUT_JSON"

echo "AIST++ test-set evaluation for $VARIANT complete. Results: $OUT_JSON"
