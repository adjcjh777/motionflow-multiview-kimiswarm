#!/usr/bin/env bash
# Variable-view MPJPE@k evaluation for the Iskakov ICCV 2019 learnable
# triangulation baseline on H36M true-GT S9/S11 validation.
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed.
# GPU 7 is usually busy with v85; this script targets GPU 6.
#
# Usage:
#   nohup bash scripts/run_iskakov_var_view_h36m_true_gt_a800.sh &

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

export CUDA_VISIBLE_DEVICES=6

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

CKPT="outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu6.pth"
OUT_DIR="outputs/variable_view_iskakov"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Iskakov H36M checkpoint not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

nohup "$PYTHON" -u experiments/eval_iskakov_mpjpe_at_k.py \
    --protocol h36m \
    --checkpoint "$CKPT" \
    --hidden_dim 32 \
    --num_subsets 50 \
    --max_frames 4000 \
    --seed 20260810 \
    --output_dir "$OUT_DIR" \
    > "$OUT_DIR/iskakov_var_view_h36m_true_gt_a800_nohup.log" 2>&1 &

PID=$!
echo "Launched Iskakov variable-view eval on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "Log: $OUT_DIR/iskakov_var_view_h36m_true_gt_a800_nohup.log"
