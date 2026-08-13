#!/usr/bin/env bash
# Local convenience wrapper: start the A800-side chain watcher that will
# sequentially launch v82, v46, v52, and v57 true-GT v2 medium trainings
# after the v81 true-GT v2 medium training finishes.
#
# The chain script lives on the A800 repo and is started via SSH with nohup
# so it survives this local terminal.
#
# Usage
# -----
#   bash scripts/launch_v82_v46_v52_v57_after_v81_a800.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

A800_HOST="${A800_HOST:-a800-D}"
A800_REPO="${A800_REPO:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20}"
CHAIN_SCRIPT="scripts/launch_v82_v46_v52_v57_after_v81.sh"
CHAIN_LOG="outputs/launch_v82_v46_v52_v57_after_v81.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting A800 chain watcher from local WSL."
log "Host: ${A800_HOST}"
log "A800 repo: ${A800_REPO}"
log "Chain log: ${A800_REPO}/${CHAIN_LOG}"

ssh -o ConnectTimeout=10 -o BatchMode=yes "${A800_HOST}" \
    "cd ${A800_REPO} && nohup bash ${CHAIN_SCRIPT} > ${CHAIN_LOG} 2>&1 &"

log "Chain watcher started on ${A800_HOST}."
log "Monitor with: ssh ${A800_HOST} \"tail -f ${A800_REPO}/${CHAIN_LOG}\""