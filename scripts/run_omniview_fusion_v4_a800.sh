#!/usr/bin/env bash
# Persistent A800-D runner for OmniMultiViewFusionV4.
#
# Picks a free GPU, activates the project venv, runs the v4 trainer under nohup,
# and restarts on non-zero exit up to a configurable number of times.  A lock
# file prevents duplicate sessions for the same experiment.
#
# Usage:
#     # Full training run (A800-D)
#     bash scripts/run_omniview_fusion_v4_a800.sh
#
#     # Smoke / dry-run (CPU-friendly, validates environment)
#     bash scripts/run_omniview_fusion_v4_a800.sh --smoke
#
#     # Show usage
#     bash scripts/run_omniview_fusion_v4_a800.sh --help
#
# Environment overrides:
#     MF_ROOT            project root (default: resolved from this script)
#     MF_VENV            path to virtualenv (default: ${MF_ROOT}/.venv)
#     MF_GPU             GPU index to use (default: auto-select)
#     MF_ALLOWED_GPUS    comma-separated list of allowed GPUs (default: 0,1,2,3,6)
#     MF_BUSY_GPUS       comma-separated list of GPUs to skip (default: 4,5,7)
#     MF_MAX_RESTARTS    max restart attempts (default: 3)
#     MF_OUTPUT_DIR      where to write .pth/.log (default: ${MF_ROOT}/outputs)
#     MF_LOCK_DIR        lock-file directory (default: ${MF_ROOT}/tmp)
#     MF_TRAINER         trainer script path (default: v4 multi-dataset trainer)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ROOT="${MF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV="${MF_VENV:-${ROOT}/.venv}"
OUTPUT_DIR="${MF_OUTPUT_DIR:-${ROOT}/outputs}"
LOCK_DIR="${MF_LOCK_DIR:-${ROOT}/tmp}"
TRAINER="${MF_TRAINER:-${ROOT}/experiments/train_omniview_fusion_v4_webbridge_multi.py}"
EXPERIMENT_NAME="${MF_EXPERIMENT_NAME:-omniview_fusion_v4}"
ALLOWED_GPUS="${MF_ALLOWED_GPUS:-6,7}"
BUSY_GPUS="${MF_BUSY_GPUS:-}"
MAX_RESTARTS="${MF_MAX_RESTARTS:-3}"

# Training hyperparameters (overridable via environment)
D="${MF_D:-128}"
RESIDUAL_HIDDEN="${MF_RESIDUAL_HIDDEN:-256}"
N_ST_LAYERS="${MF_N_ST_LAYERS:-3}"
GRAPH_NUM_LAYERS="${MF_GRAPH_NUM_LAYERS:-1}"
N_JOINT_LAYERS="${MF_N_JOINT_LAYERS:-1}"
N_HEADS="${MF_N_HEADS:-4}"
EPOCHS="${MF_EPOCHS:-30}"
BATCH_SIZE="${MF_BATCH_SIZE:-8}"
TRAIN_SAMPLES="${MF_TRAIN_SAMPLES:-4000}"
VAL_STRIDE="${MF_VAL_STRIDE:-10}"
LR="${MF_LR:-1e-3}"
MIN_VIEWS="${MF_MIN_VIEWS:-2}"
VIEW_DROPOUT_RATE="${MF_VIEW_DROPOUT_RATE:-0.1}"
VISIBILITY_LOSS_WEIGHT="${MF_VISIBILITY_LOSS_WEIGHT:-0.1}"
UNCERTAINTY_LOSS_WEIGHT="${MF_UNCERTAINTY_LOSS_WEIGHT:-0.05}"
TEMPORAL_LOSS_WEIGHT="${MF_TEMPORAL_LOSS_WEIGHT:-0.02}"
BONE_LOSS_WEIGHT="${MF_BONE_LOSS_WEIGHT:-0.05}"

LOCK_FILE="${LOCK_DIR}/${EXPERIMENT_NAME}.lock"
LOG_FILE="${OUTPUT_DIR}/${EXPERIMENT_NAME}.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    sed -n '2,/^# Environment/s/^# \{0,1\}//p' "$0"
    echo
    echo "Arguments:"
    echo "  --help     Show this help message and exit."
    echo "  --smoke    Perform a lightweight dry-run / smoke test."
    echo
    echo "Examples:"
    echo "  bash scripts/run_omniview_fusion_v4_a800.sh --smoke"
    echo "  MF_GPU=0 bash scripts/run_omniview_fusion_v4_a800.sh"
}

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "${LOG_FILE}"
}

# Pick the GPU with the lowest memory utilization among allowed GPUs,
# skipping busy GPUs and any GPU currently in use by another training process.
find_free_gpu() {
    if [[ -n "${MF_GPU:-}" ]]; then
        echo "$MF_GPU"
        return
    fi

    local gpu_list best_gpu best_util util name mem_used mem_total mem_pct
    gpu_list="${ALLOWED_GPUS//,/ }"
    best_gpu=""
    best_util=101

    for gpu in ${gpu_list}; do
        if [[ ",${BUSY_GPUS}," == *",${gpu},"* ]]; then
            continue
        fi

        # nvidia-smi may not be available in CI / WSL without GPU.
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            best_gpu="$gpu"
            break
        fi

        # utilization.gpu is reported as "0 %".
        util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "100")
        if [[ -z "$util" || "$util" == "[NotSupported]" || "$util" == "[InsufficientPermissions]" ]]; then
            util=100
        fi

        # Skip GPUs that are already occupied by other long-running processes
        # (e.g. VLLM workers) even if utilization is reported as 0%.
        mem_used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "999999")
        if [[ -n "$mem_used" && "$mem_used" =~ ^[0-9]+$ && "$mem_used" -gt 2000 ]]; then
            log "GPU $gpu has ${mem_used} MiB in use; skipping."
            continue
        fi

        if (( util < best_util )); then
            best_util="$util"
            best_gpu="$gpu"
        fi
    done

    if [[ -z "$best_gpu" ]]; then
        log "ERROR: No allowed GPU available. Allowed=${ALLOWED_GPUS}, busy=${BUSY_GPUS}."
        exit 1
    fi

    echo "$best_gpu"
}

# Acquire a lock file containing the current PID.  Fail if already held.
acquire_lock() {
    mkdir -p "$(dirname "$LOCK_FILE")"
    if [[ -f "$LOCK_FILE" ]]; then
        local pid
        pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "ERROR: Lock held by PID $pid (${LOCK_FILE}). Session already running."
            exit 1
        fi
        log "WARNING: Stale lock removed (PID $pid not alive)."
        rm -f "$LOCK_FILE"
    fi
    echo "$$" > "$LOCK_FILE"
}

release_lock() {
    if [[ -f "$LOCK_FILE" ]] && [[ "$(cat "$LOCK_FILE" 2>/dev/null)" == "$$" ]]; then
        rm -f "$LOCK_FILE"
    fi
}

# Clean up lock on exit.
trap release_lock EXIT

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SMOKE=false

for arg in "$@"; do
    case "$arg" in
        --help)
            usage
            exit 0
            ;;
        --smoke)
            SMOKE=true
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

if [[ "$SMOKE" == true ]]; then
    LOG_FILE="${OUTPUT_DIR}/${EXPERIMENT_NAME}_smoke.log"
    log "=== SMOKE / DRY-RUN mode ==="
    log "ROOT=$ROOT"
    log "VENV=$VENV"
    log "TRAINER=$TRAINER"
    log "ALLOWED_GPUS=$ALLOWED_GPUS"
    log "BUSY_GPUS=$BUSY_GPUS"

    if [[ ! -d "$VENV" ]]; then
        log "ERROR: Virtualenv not found at $VENV"
        exit 1
    fi

    GPU=$(find_free_gpu)
    log "Selected GPU: $GPU"

    if [[ ! -f "$TRAINER" ]]; then
        log "WARNING: Trainer script does not exist yet: $TRAINER"
        log "         This is expected until T08 (v4 trainer) lands."
        log "Smoke validation passed (environment + GPU selection OK)."
        exit 0
    fi

    # Attempt a trainer smoke run.
    log "Running trainer --smoke ..."
    set +e
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" \
        "${VENV}/bin/python" "$TRAINER" --smoke
    exit_code=$?
    set -e

    if (( exit_code != 0 )); then
        log "ERROR: Trainer --smoke failed with exit code $exit_code"
        exit "$exit_code"
    fi
    log "Smoke run completed successfully."
    exit 0
fi

# ---------------------------------------------------------------------------
# Normal training path
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: Virtualenv not found at $VENV" >&2
    exit 1
fi

if [[ ! -f "$TRAINER" ]]; then
    echo "ERROR: Trainer script not found at $TRAINER" >&2
    exit 1
fi

GPU=$(find_free_gpu)
export CUDA_VISIBLE_DEVICES="$GPU"
acquire_lock

log "=== Starting ${EXPERIMENT_NAME} on GPU ${GPU} (PID $$) ==="
log "VENV=$VENV"
log "TRAINER=$TRAINER"
log "LOG_FILE=$LOG_FILE"

# Optional warm-start checkpoint.
WARM_START="${MF_WARM_START:-${ROOT}/outputs/omniview_fusion_v2_d128_dense_graph_v2.pth}"
WARM_START_ARGS=()
if [[ -f "$WARM_START" ]]; then
    WARM_START_ARGS=(--warm_start "$WARM_START" --warm_start_freeze_epochs 5)
    log "Warm-start from $WARM_START"
fi

attempt=0
exit_code=0
while true; do
    attempt=$((attempt + 1))
    log "--- Training attempt $attempt / $((MAX_RESTARTS + 1)) ---"

    # nohup + unbuffered Python so the process can outlive the ssh/tmux client.
    # shellcheck disable=SC2068
    nohup "${VENV}/bin/python" -u "$TRAINER" \
        --manifest "${ROOT}/configs/splits/h36m_true_gt_standard.yaml" \
        --manifest "${ROOT}/configs/splits/mpiinf3dhp_train_val_test.yaml" \
        --d "$D" \
        --residual_hidden "$RESIDUAL_HIDDEN" \
        --n_st_layers "$N_ST_LAYERS" \
        --graph_num_layers "$GRAPH_NUM_LAYERS" \
        --n_joint_layers "$N_JOINT_LAYERS" \
        --n_heads "$N_HEADS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --train_samples "$TRAIN_SAMPLES" \
        --val_stride "$VAL_STRIDE" \
        --lr "$LR" \
        --lr_cosine \
        --lr_warmup_epochs 3 \
        --lr_min 1e-6 \
        --max_grad_norm 1.0 \
        --ema_decay 0.999 \
        --view_dropout_rate "$VIEW_DROPOUT_RATE" \
        --min_views "$MIN_VIEWS" \
        --visibility_loss_weight "$VISIBILITY_LOSS_WEIGHT" \
        --uncertainty_loss_weight "$UNCERTAINTY_LOSS_WEIGHT" \
        --temporal_loss_weight "$TEMPORAL_LOSS_WEIGHT" \
        --bone_loss_weight "$BONE_LOSS_WEIGHT" \
        "${WARM_START_ARGS[@]:+${WARM_START_ARGS[@]}}" \
        --output "${OUTPUT_DIR}/${EXPERIMENT_NAME}.pth" \
        >> "$LOG_FILE" 2>&1 &

    trainer_pid=$!
    wait "$trainer_pid"
    exit_code=$?

    if (( exit_code == 0 )); then
        log "Training finished successfully."
        break
    fi

    log "Training attempt $attempt failed with exit code $exit_code."

    if (( attempt > MAX_RESTARTS )); then
        log "ERROR: Exceeded maximum restart attempts ($MAX_RESTARTS). Giving up."
        break
    fi

    log "Restarting in 10 seconds ..."
    sleep 10
done

log "=== ${EXPERIMENT_NAME} finished with exit code ${exit_code} ==="
exit "$exit_code"
