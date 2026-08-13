#!/usr/bin/env bash
# Run the Iskakov et al. ICCV 2019 learnable-triangulation baseline on the
# H36M true-GT v2 protocol on A800.
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed.
# This script launches directly on the GPU passed via --gpu (default 6).  If you
# prefer to wait for a free GPU, set WAIT_FOR_FREE_GPU=1.
#
# Usage:
#   nohup bash scripts/run_iskakov_true_gt_v2_baseline_a800.sh \
#       > outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800_nohup.log 2>&1 &
#
# Or with an explicit GPU:
#   GPU=7 nohup bash scripts/run_iskakov_true_gt_v2_baseline_a800.sh \
#       > outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800_nohup.log 2>&1 &

set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

mkdir -p outputs/sota_baselines outputs/baselines outputs/iskakov_mpjpe_at_k_h36m_true_gt_v2

LOG="outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800.log"
exec > >(tee -a "${LOG}")
exec 2>&1

GPU="${GPU:-6}"
WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-0}"

PYTHON="${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/.venv/bin/python}"

# Validate GPU policy.
if [[ "${GPU}" != "6" && "${GPU}" != "7" ]]; then
    echo "[$(date -Iseconds)] ERROR: A800 GPU policy violation: requested GPU ${GPU}. Only 6/7 allowed." >&2
    exit 1
fi

select_free_gpu() {
    local i
    for i in 6 7; do
        local used
        used=$(nvidia-smi --id="${i}" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
        if (( used < 1000 )); then
            echo "${i}"
            return 0
        fi
    done
    return 1
}

if [[ "${WAIT_FOR_FREE_GPU}" == "1" ]]; then
    echo "[$(date -Iseconds)] Waiting for a free GPU on A800 (allowed: 6 or 7)"
    while true; do
        FREE_GPU=$(select_free_gpu) && break
        echo "[$(date -Iseconds)] No free GPU, waiting..."
        sleep 60
    done
    GPU="${FREE_GPU}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[$(date -Iseconds)] Iskakov ICCV 2019 H36M true-GT v2 baseline starting on GPU ${GPU}"

# Standard hyperparameters that reproduce the reported 23.40 mm result.
nohup "${PYTHON}" -u scripts/run_iskakov_true_gt_v2_baseline.py \
    --gpu "${GPU}" \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 8 \
    --grad_clip 1.0 \
    --seed 20260810 \
    --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --ref_max_frames 2000 \
    --eval_mpjpe_at_k \
    --num_subsets 50 \
    --max_frames 4000 \
    --device cuda \
    --log_path outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.log \
    --ckpt_path outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.pth \
    > outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800_python_nohup.log 2>&1 &

PID=$!
echo "[$(date -Iseconds)] Launched Iskakov H36M true-GT v2 baseline on GPU ${GPU} (PID: ${PID})"
echo "[$(date -Iseconds)] Main log: ${LOG}"
echo "[$(date -Iseconds)] Python nohup log: outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800_python_nohup.log"
echo "[$(date -Iseconds)] Training log: outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.log"
