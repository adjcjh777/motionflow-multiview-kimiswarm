#!/usr/bin/env bash
# Sequential H36M true-GT baseline queue (DLT -> Iskakov -> v25 -> v57 -> v80).
#
# Runs all five standard-protocol baselines one at a time on the local RTX 4090.
# The script first waits until the GPU is idle, then starts the CPU-only DLT
# baseline, followed by the four GPU training baselines (Iskakov, v25, v57, v80)
# in order.  Only one training task runs on the GPU at a time.
#
# Usage
# -----
#   bash scripts/run_all_h36m_true_gt_baselines.sh
#
#   # detached
#   nohup bash scripts/run_all_h36m_true_gt_baselines.sh \
#       > outputs/run_all_h36m_true_gt_baselines_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Environment
# ---------
#   PYTHON                 Python interpreter to use (default: python).
#   POLL_SEC               Seconds between GPU idle polls (default: 60).
#   GRACE_SEC              Idle grace period before starting GPU work (default: 90).
set -euo pipefail

PYTHON=${PYTHON:-python}
POLL_SEC=${POLL_SEC:-60}
GRACE_SEC=${GRACE_SEC:-90}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR="$REPO_ROOT/outputs"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/run_all_h36m_true_gt_baselines_$(date +%Y%m%d_%H%M%S).log"

# Per-baseline log/output files.
DLT_OUTPUT="$OUTPUT_DIR/h36m_true_gt_dlt_baseline.json"
ISKAKOV_LOG="$OUTPUT_DIR/iskakov_learnable_tri_h36m_true_gt.log"
ISKAKOV_CKPT="$OUTPUT_DIR/iskakov_learnable_tri_h36m_true_gt.pth"
V25_LOG="$OUTPUT_DIR/omniview_fusion_v25_h36m_true_gt_medium.log"
V57_LOG="$OUTPUT_DIR/omniview_fusion_v57_h36m_true_gt_medium.log"
V80_LOG="$OUTPUT_DIR/omniview_fusion_v80_h36m_true_gt_medium.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Count python processes reported by nvidia-smi.
python_gpu_count() {
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -ic "python" \
        || true
}

# Wait until no python process is using the GPU, with a grace period.
wait_for_gpu_idle() {
    log "Waiting for the local GPU to become idle..."
    while true; do
        if [[ "$(python_gpu_count)" -eq 0 ]]; then
            log "GPU reports no python processes. Grace period ${GRACE_SEC}s..."
            sleep "$GRACE_SEC"
            if [[ "$(python_gpu_count)" -eq 0 ]]; then
                log "GPU still idle after grace period. Starting baseline queue."
                break
            else
                log "A new python GPU process appeared during grace period. Continuing to wait."
            fi
        fi
        log "GPU not idle yet. Polling again in ${POLL_SEC}s..."
        sleep "$POLL_SEC"
    done
}

# Run a step and record its result.  Steps that return non-zero are logged but
# do not stop the queue, so later baselines still get a chance to run.
run_step() {
    local name="$1"
    shift
    log "------------------------------------------------------------"
    log "STARTING: $name"
    log "  command: $*"
    log "------------------------------------------------------------"
    if "$@"; then
        log "COMPLETED: $name"
        return 0
    else
        log "FAILED: $name (exit code $?)"
        return 1
    fi
}

# ---- Main queue -----------------------------------------------------------
log "H36M true-GT baseline queue started."
log "Log file: $LOG_FILE"
log "Planned runs: DLT (CPU) -> Iskakov -> v25 -> v57 -> v80"

wait_for_gpu_idle

# 1. DLT triangulation baseline (GPU-accelerated; can run while GPU is checked idle).
run_step "DLT baseline" \
    "$PYTHON" -u "$REPO_ROOT/scripts/run_h36m_true_gt_dlt_baseline.py" \
        --config "configs/splits/h36m_true_gt_standard.yaml" \
        --output "$DLT_OUTPUT" \
        --device cuda

# Remaining steps all require the GPU; each script owns the GPU exclusively.
# 2. Iskakov ICCV 2019 learnable triangulation.
run_step "Iskakov baseline" \
    "$PYTHON" -u "$REPO_ROOT/experiments/train_iskakov_baseline_shelf_campus.py" \
        --protocol h36m \
        --epochs 10 \
        --train_samples_per_epoch 4096 \
        --batch_size 8 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --patience 8 \
        --log_path "$ISKAKOV_LOG" \
        --ckpt_path "$ISKAKOV_CKPT"

# Re-check GPU idle before each learned run in case a previous step left a
# background process or failed to release memory.
wait_for_gpu_idle

# 3. v25 medium.
run_step "v25 medium" \
    bash "$REPO_ROOT/scripts/run_v25_h36m_true_gt_medium_local_4090.sh"

wait_for_gpu_idle

# 4. v57 medium.
run_step "v57 medium" \
    bash "$REPO_ROOT/scripts/run_v57_h36m_true_gt_medium.sh"

wait_for_gpu_idle

# 5. v80 medium.
run_step "v80 medium" \
    bash "$REPO_ROOT/scripts/run_v80_h36m_true_gt_medium.sh"

# ---- Summary --------------------------------------------------------------
log "------------------------------------------------------------"
log "All queued baselines have finished."
log "Artifacts:"
log "  DLT results:           $DLT_OUTPUT"
log "  Iskakov log/ckpt:      $ISKAKOV_LOG / $ISKAKOV_CKPT"
log "  v25 log:               $V25_LOG"
log "  v57 log:               $V57_LOG"
log "  v80 log:               $V80_LOG"
log "------------------------------------------------------------"
