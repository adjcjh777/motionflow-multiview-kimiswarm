#!/usr/bin/env bash
# Sequential 4090 training queue. Each job is started only after the previous one finishes.
set -euo pipefail

export EPOCHS=${EPOCHS:-20}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-500}
export BATCH_SIZE=${BATCH_SIZE:-16}

run_job() {
    local script=$1
    echo "[QUEUE] Starting $script"
    bash "$script" || echo "[QUEUE] $script failed; continuing"
}

run_job scripts/run_v25_small_local_4090.sh
run_job scripts/run_v25_v18_topk_st_local_4090.sh
run_job scripts/run_v25_v27_udp_local_4090.sh
run_job scripts/run_v25_v18_topk_v27_udp_local_4090.sh
run_job scripts/run_v25_outlier_adaptive_local_4090.sh

echo "[QUEUE] All local 4090 jobs finished"
