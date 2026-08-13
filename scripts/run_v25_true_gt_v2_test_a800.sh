#!/usr/bin/env bash
# Run v25 true-GT v2 medium H36M test-set evaluation on A800-D.
#
# Uses the best checkpoint from the v25 true-GT v2 medium run:
#   outputs/ablations/v25_true_gt_v2_medium_a800.pth
#
# Evaluates on the corrected (non-circular) H36M true-GT v2 test subjects
# S9 and S11, producing per-subject and combined MPJPE / PA-MPJPE.
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
# Default to GPU 6; override with CUDA_VISIBLE_DEVICES.
#
# The script is non-intrusive: it refuses to run if the selected GPU is
# currently occupied by another compute process. It will not preempt.
#
# Usage
# -----
#   # Default GPU 6
#   bash scripts/run_v25_true_gt_v2_test_a800.sh
#
#   # Explicit GPU
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v25_true_gt_v2_test_a800.sh
#
#   # Detached launch
#   nohup bash scripts/run_v25_true_gt_v2_test_a800.sh \
#       > outputs/eval_v25_true_gt_v2_h36m_test_a800.log 2>&1 &

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: project only uses GPUs 6/7 on A800. Default to 6, allow override.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

# Output locations.
OUT_JSON="outputs/eval_v25_true_gt_v2_h36m_test.json"
OUT_CSV="outputs/eval_v25_true_gt_v2_h36m_test.csv"

# Non-intrusive GPU check: refuse to run if the selected GPU has an compute
# process. This script will not preempt other jobs.
gpu_has_compute_apps() {
    local gpu="$1"
    local count
    count=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cv '^$' || true)
    [[ "${count:-999}" -gt 0 ]]
}

if gpu_has_compute_apps "$CUDA_VISIBLE_DEVICES"; then
    echo "ERROR: GPU ${CUDA_VISIBLE_DEVICES} is currently occupied by another process." >&2
    echo "       Please wait for it to finish or choose the other project GPU." >&2
    exit 1
fi

echo "Evaluating v25 true-GT v2 medium checkpoint on H36M test S9/S11 (GPU ${CUDA_VISIBLE_DEVICES})"
echo "  Checkpoint: outputs/ablations/v25_true_gt_v2_medium_a800.pth"
echo "  Config:     outputs/ablations/v25_true_gt_v2_medium_a800.config.json"
echo "  Output:     ${OUT_JSON}"
echo "  CSV:        ${OUT_CSV}"

$PYTHON -u scripts/eval_v25_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v25_true_gt_v2_medium_a800.pth \
    --config_json outputs/ablations/v25_true_gt_v2_medium_a800.config.json \
    --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --batch_size 8 \
    --out_json "${OUT_JSON}"

echo "Done."
echo "  JSON: ${OUT_JSON}"
echo "  CSV:  ${OUT_CSV}"
