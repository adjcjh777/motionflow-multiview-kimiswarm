#!/usr/bin/env bash
# Evaluate the AIST++-only medium fast-v2 checkpoint on the true-GT H36M S9/S11 test set.
#
# The checkpoint is produced by scripts/run_aistpp_only_medium_a800.sh (fast v2 variant).
# The trainer saves the final state as aistpp_only_medium_a800_fast_v2_final.pth; a
# symlink aistpp_only_medium_a800_fast_v2.pth -> ..._final.pth is created so that
# downstream references resolve.  Run this script only after training finishes and
# the checkpoint/config JSON are present.
#
# Usage
# -----
#   bash scripts/eval_aistpp_only_on_h36m_test.sh
#
#   # Use a different GPU
#   CUDA_VISIBLE_DEVICES=7 bash scripts/eval_aistpp_only_on_h36m_test.sh
set -euo pipefail

# Pin to the A800-D repo where the checkpoint lives.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

CKPT="outputs/ablations/aistpp_only_medium_a800_fast_v2.pth"
CONFIG="outputs/ablations/aistpp_only_medium_a800_fast_v2.config.json"
OUT_JSON="outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT" >&2
    echo "       Wait for AIST++-only training to finish, then re-run." >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config JSON not found: $CONFIG" >&2
    exit 1
fi

echo "Evaluating AIST++-only checkpoint on H36M true-GT S9/S11 (GPU ${CUDA_VISIBLE_DEVICES})"

$PYTHON -u scripts/eval_v25_true_gt_h36m_test.py \
    --checkpoint "$CKPT" \
    --config_json "$CONFIG" \
    --s9 data/h36m_true_gt/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --clip_len 13 \
    --batch_size 8 \
    --val_stride 1 \
    --out_json "$OUT_JSON"

echo "Done. Results: $OUT_JSON"
