#!/usr/bin/env bash
# watchdog: monitor the local RTX 4090 WSL training run and auto-eval new checkpoints.
#
# Responsibilities:
#   * log GPU memory/temperature at each poll
#   * detect when the 4090 v2 checkpoint has been updated
#   * trigger the 4090-only eval once per checkpoint, but only when no other
#     eval (local or A800-D) is already running.
#
# Usage:
#   # daemon mode (poll every 60 s)
#   bash scripts/monitor_4090_v2_and_auto_eval.sh
#
#   # single-shot dry-run (no eval launched, prints to stdout)
#   bash scripts/monitor_4090_v2_and_auto_eval.sh --dry-run --once
#
# Options:
#   --lock-dir <dir>  Use <dir> for all lock files (useful for tests/cron).
#
# Environment overrides:
#   MF_4090_CKPT      - checkpoint to watch [outputs/omniview_fusion_v2_d128_dense_graph_v2.pth]
#   MF_4090_LOG       - monitor log file     [outputs/monitor_4090_v2.log]
#   MF_4090_LOCK      - watchdog lock file   [/tmp/motionflow_4090_monitor.lock]
#   MF_4090_INTERVAL  - poll interval in sec   [60]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CKPT="${MF_4090_CKPT:-${ROOT}/outputs/omniview_fusion_v2_d128_dense_graph_v2.pth}"
MONITOR_LOG="${MF_4090_LOG:-${ROOT}/outputs/monitor_4090_v2.log}"
LOCK_FILE="${MF_4090_LOCK:-/tmp/motionflow_4090_monitor.lock}"
AUTO_EVAL_LOCK="${MF_4090_AUTO_EVAL_LOCK:-/tmp/motionflow_4090_auto_eval.lock}"
A800_LOCK="${MF_4090_A800_LOCK:-/tmp/motionflow_a800_eval.lock}"
GENERIC_LOCK="${MF_4090_GENERIC_LOCK:-/tmp/motionflow_auto_eval.lock}"
INTERVAL_SEC="${MF_4090_INTERVAL:-60}"
LOCK_DIR=""
DRY_RUN=0
ONCE=0

# Eval outputs (mirrors scripts/auto_eval_when_ready.sh)
JSON_OUT="${ROOT}/outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.json"
CSV_OUT="${ROOT}/outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.csv"
LOG_OUT="${ROOT}/outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.log"
DATASET="${ROOT}/data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"

V2_ARGS="--d 128 --residual_hidden 256 --n_st_layers 3 --graph_num_layers 1 --n_joint_layers 1"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --dry-run   Do not launch eval; log the action that would be taken.
  --once      Run one polling cycle and exit (useful from cron).
  --interval  Poll interval in seconds (default: $INTERVAL_SEC).
  --ckpt      Checkpoint path to watch (default: $CKPT).
  --log       Path to monitor log (default: $MONITOR_LOG).
  --lock-dir  Directory for all lock files (default: /tmp).
  -h, --help  Show this help.
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --once) ONCE=1; shift ;;
        --interval) INTERVAL_SEC="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --log) MONITOR_LOG="$2"; shift 2 ;;
        --lock-dir) LOCK_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [ -n "$LOCK_DIR" ]; then
    mkdir -p "$LOCK_DIR"
    LOCK_FILE="$LOCK_DIR/motionflow_4090_monitor.lock"
    AUTO_EVAL_LOCK="$LOCK_DIR/motionflow_4090_auto_eval.lock"
    A800_LOCK="$LOCK_DIR/motionflow_a800_eval.lock"
    GENERIC_LOCK="$LOCK_DIR/motionflow_auto_eval.lock"
fi

mkdir -p "$(dirname "$MONITOR_LOG")"
mkdir -p "$(dirname "$CKPT")"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date -Iseconds)] $*"
    echo "$msg" | tee -a "$MONITOR_LOG"
}

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid
        pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
        if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
            echo "Watchdog already running (pid=$pid)." >&2
            return 1
        fi
        # Stale lock.
        rm -f "$LOCK_FILE"
    fi
    echo "$$" > "$LOCK_FILE"
    return 0
}

cleanup() {
    rm -f "$LOCK_FILE"
}

check_any_eval_lock() {
    # Prevent duplicate or overlapping evals with the generic auto-eval script
    # and any A800-D cron eval.
    if [ -f "$GENERIC_LOCK" ] || [ -f "$A800_LOCK" ] || [ -f "$AUTO_EVAL_LOCK" ]; then
        return 0
    fi
    return 1
}

log_gpu_status() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=timestamp,name,temperature.gpu,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | while IFS= read -r line; do
            log "gpu_status: $line"
        done
    else
        log "gpu_status: nvidia-smi not available"
    fi
}

training_is_running() {
    pgrep -f "train_omniview_fusion_v2" >/dev/null 2>&1 || pgrep -f "train_omniview_fusion" >/dev/null 2>&1
}

run_4090_eval() {
    log "launching 4090 eval: $CKPT"
    python "${ROOT}/experiments/eval_omniview_fusion_v2_mpiinf3dhp.py" \
        $V2_ARGS \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --run_robustness \
        --run_variable_views \
        --out_json "$JSON_OUT" \
        --out_csv "$CSV_OUT" \
        >"$LOG_OUT" 2>&1
}

# Launch eval in the background while holding a lock file.  Only one eval can
# run at a time, and the lock is removed when the eval finishes so future
# watchdog cycles can detect that no eval is running.
spawn_4090_eval() {
    local done_file="$1"
    (
        echo "$$" > "$AUTO_EVAL_LOCK"
        if run_4090_eval; then
            touch "$done_file"
            log "4090 eval completed: $JSON_OUT"
        else
            log "4090 eval failed, see $LOG_OUT"
        fi
        rm -f "$AUTO_EVAL_LOCK"
    ) &
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if ! acquire_lock; then
    exit 0
fi
trap cleanup EXIT

log "monitor started (pid=$$) -- ckpt=$CKPT interval=${INTERVAL_SEC}s dry_run=$DRY_RUN"

while true; do
    log_gpu_status

    if training_is_running; then
        log "training: running"
    else
        log "training: not detected"
    fi

    if [ -f "$CKPT" ]; then
        DONE_FILE="${CKPT}.eval_done"
        CKPT_MTIME=$(date -r "$CKPT" +%s 2>/dev/null || stat -c %Y "$CKPT" 2>/dev/null || echo 0)
        TRIGGER=0
        if [ ! -f "$DONE_FILE" ]; then
            TRIGGER=1
        else
            DONE_MTIME=$(date -r "$DONE_FILE" +%s 2>/dev/null || stat -c %Y "$DONE_FILE" 2>/dev/null || echo 0)
            if [ "$CKPT_MTIME" -gt "$DONE_MTIME" ]; then
                TRIGGER=1
            fi
        fi

        if [ "$TRIGGER" -eq 1 ]; then
            log "new checkpoint detected: $CKPT"
            if check_any_eval_lock; then
                log "eval lock present (auto-eval or A800-D); deferring 4090 eval"
            else
                if [ "$DRY_RUN" -eq 1 ]; then
                    log "DRY-RUN: would launch 4090 eval for $CKPT"
                else
                    spawn_4090_eval "$DONE_FILE"
                fi
            fi
        else
            log "checkpoint up-to-date (eval_done: $DONE_FILE)"
        fi
    else
        log "checkpoint not found: $CKPT"
    fi

    if [ "$ONCE" -eq 1 ]; then
        log "single-shot complete; exiting"
        break
    fi

    sleep "$INTERVAL_SEC"
done
