#!/usr/bin/env bash
# Run VoxelPose baseline on the corrected H36M true-GT protocol.
#
# This is a thin wrapper around
# scripts/sota_baselines/prepare_voxelpose_h36m.sh. It performs the
# safety checks, creates the log directory, and redirects output to a log
# file so the run can be launched with nohup.
#
# Do not run this script automatically. Launch it manually only when the
# local RTX 4090 is free and no other training/evaluation task is running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Safety guard: never run on A800-D / read-only mount.
HOSTNAME=$(hostname 2>/dev/null || uname -n)
if [[ "${HOSTNAME}" == "a800-D"* ]] || [[ -d /mnt/nvme0n1/zhangzy/projects ]]; then
    echo "ERROR: This script must not run on A800-D (read-only)." >&2
    exit 1
fi

# Safety guard: do not start if the GPU is busy.
if ! bash "${SCRIPT_DIR}/sota_baselines/check_gpu_free.sh"; then
    echo "GPU is not free. Aborting." >&2
    exit 1
fi

mkdir -p outputs/sota_baselines

LOG="outputs/sota_baselines/voxelpose_h36m_true_gt_run.log"
echo "Starting VoxelPose H36M true-GT baseline. Log: ${LOG}"

bash "${SCRIPT_DIR}/sota_baselines/prepare_voxelpose_h36m.sh" > "${LOG}" 2>&1

echo "VoxelPose H36M true-GT baseline complete. See ${LOG}"
