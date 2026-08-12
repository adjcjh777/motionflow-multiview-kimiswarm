#!/usr/bin/env bash
# Launch AIST++ full-medium training on A800.
#
# Uses the two currently free training GPUs (4 and 6) for v25 and v80.
# v57 AIST++ medium can be launched manually once GPU 5 (H36M v57 re-run) is free.
#
# Usage
# -----
#   bash scripts/launch_aistpp_medium_a800.sh
#
# To run detached:
#   nohup bash scripts/launch_aistpp_medium_a800.sh > outputs/launch_aistpp_medium_a800.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=== AIST++ medium A800 launch ==="
log "Repo: ${REPO}"
log "Launching v25 on GPU 4 and v80 on GPU 6"

# v25 on GPU 4
log "Starting v25 AIST++ medium on GPU 4"
CUDA_VISIBLE_DEVICES=4 nohup bash "${SCRIPT_DIR}/run_v25_aistpp_full_medium_a800.sh" > outputs/v25_aistpp_full_medium_gpu4_nohup.log 2>&1 &
V25_PID=$!

# v80 on GPU 6
log "Starting v80 AIST++ medium on GPU 6"
CUDA_VISIBLE_DEVICES=6 nohup bash "${SCRIPT_DIR}/run_v80_aistpp_full_medium_a800.sh" > outputs/v80_aistpp_full_medium_gpu6_nohup.log 2>&1 &
V80_PID=$!

log "Launched v25 (PID ${V25_PID}) and v80 (PID ${V80_PID})"
log "Tail logs:"
log "  tail -f outputs/v25_aistpp_full_medium_gpu4_nohup.log"
log "  tail -f outputs/v80_aistpp_full_medium_gpu6_nohup.log"
log "Per-run logs:"
log "  outputs/ablations/v25_aistpp_full_medium_a800.log"
log "  outputs/ablations/v80_aistpp_full_medium_a800.log"
