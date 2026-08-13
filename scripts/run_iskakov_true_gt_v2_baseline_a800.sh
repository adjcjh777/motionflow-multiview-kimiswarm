#!/usr/bin/env bash
# Run the Iskakov et al. ICCV 2019 learnable-triangulation baseline on the
# H36M true-GT v2 protocol on A800, queuing until a project GPU is free.
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed.
# The script polls both GPUs and launches on the first one whose utilization,
# memory use, and compute-process count are all below threshold. This makes it
# safe to start while another training run (e.g. v81) is still active.
#
# Usage:
#   nohup bash scripts/run_iskakov_true_gt_v2_baseline_a800.sh \
#       > outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800_nohup.log 2>&1 &
#
# The script waits for a free GPU, runs 10 epochs, then prints and records the
# S9/S11 combined MPJPE.

set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

mkdir -p outputs/sota_baselines outputs/baselines outputs/iskakov_mpjpe_at_k_h36m_true_gt_v2

LOG="outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800.log"
exec > >(tee -a "${LOG}")
exec 2>&1

ALLOWED_GPUS=(6 7)
POLL_SEC=60

PYTHON="${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/.venv/bin/python}"

# Return the first GPU index (6 or 7) whose utilization and memory are below
# threshold and which has no compute processes. Empty string if none free.
find_free_gpu() {
    local gpu util mem procs
    for gpu in "${ALLOWED_GPUS[@]}"; do
        read -r util mem <<< "$(nvidia-smi --id=${gpu} --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')"
        procs="$(nvidia-smi --id=${gpu} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"

        util="${util//%/}"; util="${util:-100}"
        mem="${mem//MiB/}"; mem="${mem:-999999}"
        procs="${procs// /}"

        if [[ "${util}" =~ ^[0-9]+$ && "${mem}" =~ ^[0-9]+$ && "${procs}" =~ ^[0-9]+$ ]] \
           && [[ "${util}" -lt 10 ]] \
           && [[ "${mem}" -lt 1000 ]] \
           && [[ "${procs}" -eq 0 ]]; then
            echo "${gpu}"
            return 0
        fi
    done
    echo ""
}

# Wait until a project GPU becomes free. Do not kill or disturb running jobs.
echo "[$(date -Iseconds)] Iskakov ICCV 2019 H36M true-GT v2 baseline A800 queue monitor starting"
echo "[$(date -Iseconds)] Waiting for a free project GPU (allowed: 6 or 7)..."
while true; do
    FREE_GPU=$(find_free_gpu)
    [[ -n "${FREE_GPU}" ]] && break
    echo "[$(date -Iseconds)] No free project GPU (GPU 6/7); polling in ${POLL_SEC}s..."
    sleep "${POLL_SEC}"
done

echo "[$(date -Iseconds)] GPU ${FREE_GPU} is free; launching Iskakov ICCV 2019 H36M true-GT v2 baseline"

export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LOG_PATH="outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.log"
CKPT_PATH="outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.pth"
METRICS_PATH="outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.metrics.json"

# Standard hyperparameters that reproduce the reported 23.40 mm result.
"${PYTHON}" -u scripts/run_iskakov_true_gt_v2_baseline.py \
    --gpu "${FREE_GPU}" \
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
    --log_path "${LOG_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --metrics_json "${METRICS_PATH}" \
    > outputs/sota_baselines/run_iskakov_true_gt_v2_baseline_a800_python_nohup.log 2>&1

EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "[$(date -Iseconds)] ERROR: Python training/eval failed with exit code ${EXIT_CODE}" >&2
    exit ${EXIT_CODE}
fi

# Print and record the S9/S11 combined MPJPE.
if [[ -f "${METRICS_PATH}" ]]; then
    COMBINED_MPJPE=$(${PYTHON} -c "import json,sys; print(json.load(open('${METRICS_PATH}'))['s9_s11_combined_mpjpe_mm'])")
    echo "[$(date -Iseconds)] S9/S11 combined MPJPE: ${COMBINED_MPJPE} mm"
    echo "[$(date -Iseconds)] S9/S11 combined MPJPE: ${COMBINED_MPJPE} mm" >> outputs/sota_baselines/iskakov_true_gt_v2_combined_mpjpe.txt
else
    echo "[$(date -Iseconds)] Warning: metrics JSON not found at ${METRICS_PATH}; cannot report S9/S11 combined MPJPE" >&2
fi

echo "[$(date -Iseconds)] Iskakov H36M true-GT v2 baseline finished."
echo "[$(date -Iseconds)] Main log: ${LOG}"
echo "[$(date -Iseconds)] Training log: ${LOG_PATH}"
echo "[$(date -Iseconds)] Metrics JSON: ${METRICS_PATH}"
