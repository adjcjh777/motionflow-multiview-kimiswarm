#!/usr/bin/env bash
# Safety check before launching A800 training jobs.
# SSH to a800-D, inspect GPU occupancy, and return 0 only if at least one A800
# GPU is free (low memory usage and no training process) and no conflicting
# training is running on it.
set -euo pipefail

SSH_HOST="a800-D"
FREE_MEMORY_THRESHOLD_MB=5000

# Command-line substrings that indicate a training run is using a GPU.
TRAINING_PATTERNS=(
    "train_omniview_fusion"
    "omniview_fusion"
)

usage() {
    cat >&2 <<EOF
Usage: $0 [-h|--help]

SSH to ${SSH_HOST}, inspect GPU occupancy, and return 0 only if at least one
GPU is free (low memory and no training process) and no conflicting training
is running on it.

Exit codes:
  0  safe to launch
  1  SSH or nvidia-smi query failed
  3  no free GPU available
EOF
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help) usage; exit 0 ;;
    esac
done

a800_ssh() {
    ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSH_HOST}" "$@" </dev/null
}

# Verify SSH works.
if ! a800_ssh "true" >/dev/null 2>&1; then
    echo "ERROR: Cannot reach ${SSH_HOST} via SSH." >&2
    exit 1
fi

# Build a pipe-friendly grep pattern from the training markers.
PATTERN_RE="$(printf '%s\n' "${TRAINING_PATTERNS[@]}" | paste -sd '|')"

# Query per-GPU memory and inspect the full command line of every process on
# each GPU to detect active training runs.
gpu_info=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSH_HOST}" \
    "PATTERN_RE='${PATTERN_RE}' bash -s" <<'REMOTE_EOF'
set -euo pipefail
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while IFS=',' read -r idx mem_used; do
    idx=$(echo "$idx" | xargs)
    mem_used=$(echo "$mem_used" | xargs)
    has_training=0
    if [ -n "$idx" ]; then
        while IFS=',' read -r pid; do
            pid=$(echo "$pid" | xargs)
            if [ -n "$pid" ]; then
                cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || true)
                if echo "$cmd" | grep -qiE "$PATTERN_RE"; then
                    has_training=1
                fi
            fi
        done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=$idx 2>/dev/null || true)
    fi
    echo "$idx,$mem_used,$has_training"
done
REMOTE_EOF
)

if [[ -z "${gpu_info}" ]]; then
    echo "ERROR: Failed to query GPU status on ${SSH_HOST}." >&2
    exit 1
fi

free_gpus=()
while IFS=',' read -r idx mem_used has_training; do
    idx=$(echo "${idx}" | xargs)
    mem_used=$(echo "${mem_used}" | xargs)
    has_training=$(echo "${has_training}" | xargs)
    [[ -z "${idx}" ]] && continue

    if [[ "${has_training}" -eq 1 ]]; then
        continue
    fi

    if [[ "${mem_used}" -lt "${FREE_MEMORY_THRESHOLD_MB}" ]]; then
        free_gpus+=("${idx}")
    fi
done <<< "${gpu_info}"

if [[ ${#free_gpus[@]} -eq 0 ]]; then
    echo "ERROR: No free GPU on ${SSH_HOST} (threshold: ${FREE_MEMORY_THRESHOLD_MB} MB)." >&2
    echo "GPU status (index,memory.used,has_training):" >&2
    echo "${gpu_info}" >&2
    exit 3
fi

echo "OK: GPU(s) ${free_gpus[*]} appear free on ${SSH_HOST}."
exit 0
