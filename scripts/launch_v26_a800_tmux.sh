#!/usr/bin/env bash
# Launch v26 (temporal multi-view geometry fusion) on A800-D via tmux.
#
# Usage:
#   bash scripts/launch_v26_a800_tmux.sh [GPU ...]
#
# If no GPU is supplied, the script auto-selects the least-utilised free GPU.
# If one or more GPUs are supplied, a tmux session is started on each.
#
# Examples:
#   bash scripts/launch_v26_a800_tmux.sh 4
#   bash scripts/launch_v26_a800_tmux.sh 4 6
#
# Environment overrides:
#   MF_GPU              GPU index to use (default: auto-select a free GPU)
#   MF_ALLOWED_GPUS     comma-separated GPUs to consider (default: 0,1,2,3,4,5,6,7)
#   MF_BUSY_GPUS        comma-separated GPUs to skip  (default: empty)
set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
PYTHON_BIN="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python"
PYTHONPATH="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"

ALLOWED_GPUS="${MF_ALLOWED_GPUS:-0,1,2,3,4,5,6,7}"
BUSY_GPUS="${MF_BUSY_GPUS:-}"
MEMORY_THRESHOLD_MIB="${MF_MEMORY_THRESHOLD_MIB:-2000}"

find_free_gpu() {
    local gpu_list best_gpu best_util util mem_used
    gpu_list="${ALLOWED_GPUS//,/ }"
    best_gpu=""
    best_util=101

    for gpu in ${gpu_list}; do
        if [[ ",${BUSY_GPUS}," == *",${gpu},"* ]]; then
            continue
        fi

        if ! command -v nvidia-smi >/dev/null 2>&1; then
            best_gpu="$gpu"
            break
        fi

        util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "100")
        if [[ -z "$util" || "$util" == "[NotSupported]" || "$util" == "[InsufficientPermissions]" ]]; then
            util=100
        fi

        mem_used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "999999")
        if [[ -n "$mem_used" && "$mem_used" =~ ^[0-9]+$ && "$mem_used" -gt "$MEMORY_THRESHOLD_MIB" ]]; then
            continue
        fi

        if (( util < best_util )); then
            best_util="$util"
            best_gpu="$gpu"
        fi
    done

    echo "$best_gpu"
}

launch_one() {
    local gpu=$1
    local name="v26_temporal_geometry_fusion_small_gpu${gpu}"
    local output="outputs/omniview_fusion_v26_temporal_geometry_fusion_small_gpu${gpu}.pth"
    local log="outputs/omniview_fusion_v26_temporal_geometry_fusion_small_gpu${gpu}.log"
    local script="scripts/run_v26_temporal_geometry_fusion_a800_small.sh"

    tmux kill-session -t "$name" 2>/dev/null || true
    tmux new-session -d -s "$name" \
        "export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PYTHONPATH; export PYTHON=$PYTHON_BIN; export OUTPUT=$output; export LOG=$log; bash $script"

    echo "Launched v26 small on GPU $gpu as tmux session $name"
}

GPUS=()

if [[ -n "${MF_GPU:-}" ]]; then
    GPUS=("$MF_GPU")
else
    for arg in "$@"; do
        GPUS+=("$arg")
    done

    if [[ ${#GPUS[@]} -eq 0 ]]; then
        free_gpu=$(find_free_gpu)
        if [[ -z "$free_gpu" ]]; then
            echo "ERROR: No free GPU available. Allowed=${ALLOWED_GPUS}, busy=${BUSY_GPUS}." >&2
            exit 1
        fi
        GPUS=("$free_gpu")
    fi
fi

for gpu in "${GPUS[@]}"; do
    launch_one "$gpu"
done

tmux list-sessions
