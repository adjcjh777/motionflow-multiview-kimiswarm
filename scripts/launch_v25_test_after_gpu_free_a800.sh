#!/usr/bin/env bash
# A800-local watcher: wait until GPU 6 or 7 is free, then run the v25 true-GT v2
# medium test-set evaluation.
#
# Usage (on A800):
#   nohup bash scripts/launch_v25_test_after_gpu_free_a800.sh > outputs/launch_v25_test_after_gpu_free_a800.log 2>&1 &

set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
V25_TEST_SCRIPT="scripts/run_v25_true_gt_v2_test_a800.sh"
POLL_SEC=60

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Prefer GPU 6, fallback to 7.
select_free_gpu() {
    for idx in 6 7; do
        local proc_count
        proc_count=$(nvidia-smi --id="${idx}" --query-compute-apps=pid --format=csv,noheader | grep -v "^[[:space:]]*$" | wc -l)
        if [[ "${proc_count}" -eq 0 ]]; then
            echo "${idx}"
            return 0
        fi
    done
    return 1
}

cd "${REPO}"

log "A800 watcher started. Waiting for GPU 6/7 to become free."

FREE_GPU=""
while true; do
    if FREE_GPU="$(select_free_gpu)"; then
        break
    fi
    log "No free GPU 6/7; waiting ${POLL_SEC}s..."
    sleep "${POLL_SEC}"
done

log "GPU ${FREE_GPU} is free. Launching v25 test-set eval."

# Modify eval script to use the free GPU.
sed -i "s/CUDA_VISIBLE_DEVICES=.*/CUDA_VISIBLE_DEVICES=${FREE_GPU}/" "${V25_TEST_SCRIPT}"
bash "${V25_TEST_SCRIPT}"

log "v25 test-set eval launched on GPU ${FREE_GPU}. Exiting."
