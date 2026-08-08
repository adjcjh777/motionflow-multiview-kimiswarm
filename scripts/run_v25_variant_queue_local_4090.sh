#!/usr/bin/env bash
# v25 local 4090 variant queue.
# Runs a sequence of v25-related small-scale experiments back-to-back.
# Intended to be launched via nohup so it survives shell disconnect.
set -euo pipefail

mkdir -p outputs

LOG_DIR="outputs/v25_variant_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Map experiment tag to the runner script.
declare -a RUNNERS=(
    "baseline:scripts/run_v25_small_local_4090.sh"
    "v18_topk_st:scripts/run_v25_v18_topk_st_local_4090.sh"
    "v27_udp:scripts/run_v25_v27_udp_local_4090.sh"
    "v18_topk_v27_udp:scripts/run_v25_v18_topk_v27_udp_local_4090.sh"
    "outlier_adaptive:scripts/run_v25_outlier_adaptive_local_4090.sh"
    "v28:scripts/run_v25_v28_local_4090.sh"
)

for entry in "${RUNNERS[@]}"; do
    tag="${entry%%:*}"
    script="${entry#*:}"
    echo "[$(date)] === Starting $tag with $script ===" | tee -a "$LOG_DIR/queue.log"
    if bash "$script" > "$LOG_DIR/${tag}.log" 2>&1; then
        echo "[$(date)] $tag completed" | tee -a "$LOG_DIR/queue.log"
    else
        echo "[$(date)] $tag FAILED (exit $?)" | tee -a "$LOG_DIR/queue.log"
    fi
done

echo "[$(date)] All v25 variant runs finished. Summary:" | tee -a "$LOG_DIR/queue.log"
for tag in baseline v18_topk_st v27_udp v18_topk_v27_udp outlier_adaptive v28; do
    best=$(grep -oP 'val_MPJPE=\K[0-9.]+' "$LOG_DIR/${tag}.log" 2>/dev/null | head -1 || echo "N/A")
    echo "  $tag best val_MPJPE: $best" | tee -a "$LOG_DIR/queue.log"
done
