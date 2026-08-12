#!/usr/bin/env bash
# Poll the A800 host and return the first free GPU index among 6-7.
#
# A GPU is considered free when:
#   - utilization.gpu == 0 %
#   - memory.used < FREE_MEMORY_THRESHOLD_MB (default 5000 MiB)
#
# Usage:
#   scripts/wait_for_a800_gpu.sh [options]
#
# Options:
#   -t, --timeout SECONDS   Exit with error after SECONDS (default 0 = no timeout).
#   -i, --interval SECONDS  Polling interval in seconds (default 10).
#   -m, --memory MB         Memory threshold in MiB (default 5000).
#
# Exit codes:
#   0  success; prints the free GPU index (4-7) to stdout.
#   1  SSH/nvidia-smi failure or timeout reached without finding a free GPU.
set -euo pipefail

SSH_HOST="a800-D"
FREE_MEMORY_THRESHOLD_MB=${FREE_MEMORY_THRESHOLD_MB:-5000}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-10}
TIMEOUT_SEC=${TIMEOUT_SEC:-0}

TARGET_GPUS="6 7"

usage() {
    cat >&2 <<EOF
Usage: $0 [options]

Poll ${SSH_HOST} and return the first free GPU index among ${TARGET_GPUS}.

Options:
  -t, --timeout SECONDS   Exit with error after SECONDS (default 0 = no timeout).
  -i, --interval SECONDS  Polling interval in seconds (default ${POLL_INTERVAL_SEC}).
  -m, --memory MB         Memory threshold in MiB (default ${FREE_MEMORY_THRESHOLD_MB}).
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--timeout)
            TIMEOUT_SEC="$2"
            shift 2
            ;;
        -i|--interval)
            POLL_INTERVAL_SEC="$2"
            shift 2
            ;;
        -m|--memory)
            FREE_MEMORY_THRESHOLD_MB="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Verify SSH works.
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSH_HOST}" "true" </dev/null >/dev/null 2>&1; then
    echo "ERROR: Cannot reach ${SSH_HOST} via SSH." >&2
    exit 1
fi

START_TIME=${SECONDS}

poll_free_gpu() {
    ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSH_HOST}" \
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits" </dev/null 2>/dev/null
}

is_elapsed() {
    if [[ "${TIMEOUT_SEC}" -gt 0 && $((SECONDS - START_TIME)) -ge "${TIMEOUT_SEC}" ]]; then
        return 0
    fi
    return 1
}

while true; do
    if is_elapsed; then
        echo "ERROR: Timeout reached after ${TIMEOUT_SEC}s without finding a free GPU among ${TARGET_GPUS}." >&2
        exit 1
    fi

    gpu_info=$(poll_free_gpu) || {
        echo "ERROR: Failed to query GPU status on ${SSH_HOST}." >&2
        exit 1
    }

    free_gpu=""
    while IFS=',' read -r idx util mem; do
        idx=$(echo "${idx}" | xargs)
        util=$(echo "${util}" | xargs)
        mem=$(echo "${mem}" | xargs)

        # Only consider GPUs 4-7.
        if [[ ! " ${TARGET_GPUS} " =~ \ ${idx}\  ]]; then
            continue
        fi

        # Remove % from utilization, if present.
        util_num="${util%%%}"

        if [[ "${util_num}" == "0" && "${mem}" -lt "${FREE_MEMORY_THRESHOLD_MB}" ]]; then
            free_gpu="${idx}"
            break
        fi
    done <<< "${gpu_info}"

    if [[ -n "${free_gpu}" ]]; then
        echo "${free_gpu}"
        exit 0
    fi

    sleep "${POLL_INTERVAL_SEC}"
done
