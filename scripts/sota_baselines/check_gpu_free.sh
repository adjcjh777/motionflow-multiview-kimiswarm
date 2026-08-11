#!/usr/bin/env bash
# Helper: return 0 if the local RTX 4090 is idle, non-zero otherwise.
# This is used by the SOTA baseline prep scripts to avoid launching GPU work
# while agent-51 or another training run is in progress.
set -euo pipefail

# Detect if we are on the A800-D remote (read-only). Never run baselines there.
HOSTNAME=$(hostname 2>/dev/null || uname -n)
if [[ "${HOSTNAME}" == "a800-D"* ]] || [[ -d /mnt/nvme0n1/zhangzy/projects ]]; then
    echo "ERROR: This script must not run on A800-D (read-only)." >&2
    exit 1
fi

# nvidia-smi must be available.
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Cannot verify GPU availability." >&2
    exit 1
fi

# On Windows nvidia-smi --query-compute-apps lists every process that has ever
# touched the display driver (including dwm/explorer). We therefore decide
# "busy" by actual GPU memory usage and by python processes that hold memory.
MEMORY_USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
# Sum memory across all GPUs (this host has one RTX 4090, but keep robust).
TOTAL_MEMORY=0
for m in ${MEMORY_USED_MIB}; do
    TOTAL_MEMORY=$((TOTAL_MEMORY + m))
done

# Threshold: if more than ~2 GiB is in use, assume a training run is active.
THRESHOLD=2048
if [[ "${TOTAL_MEMORY}" -gt "${THRESHOLD}" ]]; then
    echo "GPU memory in use: ${TOTAL_MEMORY} MiB (> ${THRESHOLD} MiB threshold)." >&2
    echo "GPU is busy. Wait for the current run to finish." >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv >&2
    exit 2
fi

# Also refuse if any python process is reported with non-N/A GPU memory.
PYTHON_GPU_MEM=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
    awk -F',' 'tolower($2) ~ /python/ && $3 !~ /N\/A/ && $3+0 > 0 {print}')
if [[ -n "${PYTHON_GPU_MEM}" ]]; then
    echo "GPU is busy; python process(es) hold GPU memory:" >&2
    echo "${PYTHON_GPU_MEM}" >&2
    exit 2
fi

echo "GPU is idle. Safe to start a baseline."
exit 0
