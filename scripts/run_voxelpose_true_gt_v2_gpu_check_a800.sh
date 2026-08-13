#!/usr/bin/env bash
# GPU sanity check for VoxelPose true-GT v2: run exactly one epoch on the
# first free project GPU (6 or 7).  Used to confirm the first GPU epoch does
# NOT raise "Invalid device id" before committing a full 10-epoch run.
#
# Usage:
#   ssh a800-D 'bash /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/scripts/run_voxelpose_true_gt_v2_gpu_check_a800.sh'
#
# The script blocks until GPU 6 or 7 is free, then exits after one epoch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONVERT_CONFIG="${REPO_ROOT}/configs/sota_baselines/voxelpose_h36m_true_gt_v2_prep.yaml"
RUN_CONFIG="${REPO_ROOT}/configs/sota_baselines/voxelpose_h36m_true_gt_v2_gpu_check.yaml"

VOXELPOSE_DIR="${REPO_ROOT}/models/voxelpose-pytorch"
CONDA_EXE="${CONDA_EXE:-/mnt/nvme0n1p1/zhangzy/tools/miniconda3/bin/conda}"
VENV_NAME="${VENV_NAME:-voxelpose_py38_pt112}"

LOG_DIR="${REPO_ROOT}/outputs/sota_baselines"
LOG_FILE="${LOG_DIR}/voxelpose_true_gt_v2_gpu_check_a800.log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "[$(date -Iseconds)] VoxelPose true-GT v2 GPU check starting"
echo "repo root: ${REPO_ROOT}"
echo "run config: ${RUN_CONFIG}"

cd "${REPO_ROOT}"

# Validate conda environment.
if ! "${CONDA_EXE}" env list | grep -qE "^\s*${VENV_NAME}\s+"; then
    echo "ERROR: conda env '${VENV_NAME}' not found." >&2
    exit 1
fi

# Ensure converted data exists.
if [[ ! -f "tmp/sota_baselines/voxelpose_data_v2/h36m_true_gt_annotations.pkl" ]]; then
    echo "Converting common baseline format to VoxelPose input ..."
    "${CONDA_EXE}" run -n "${VENV_NAME}" python \
        scripts/sota_baselines/convert_to_voxelpose_format.py \
        --config "${CONVERT_CONFIG}"
fi

# Ensure upstream VoxelPose repo is present and overlay is applied.
if [[ ! -f "${VOXELPOSE_DIR}/run/train_3d.py" ]]; then
    echo "ERROR: VoxelPose repo not found at ${VOXELPOSE_DIR}" >&2
    exit 1
fi
cp "${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay/h36m_true_gt.py" \
    "${VOXELPOSE_DIR}/lib/dataset/h36m_true_gt.py"
cp "${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay/__init__.py" \
    "${VOXELPOSE_DIR}/lib/dataset/__init__.py"
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay/patch_voxelpose_function.py" \
    "${VOXELPOSE_DIR}/lib/core/function.py"
mkdir -p "${VOXELPOSE_DIR}/output" "${VOXELPOSE_DIR}/log"

# Wait until a project GPU (6 or 7) is free.
select_free_gpu() {
    local i
    for i in 6 7; do
        local used
        used=$(nvidia-smi --id="${i}" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
        if (( used < 1000 )); then
            echo "${i}"
            return 0
        fi
    done
    return 1
}

echo "Waiting for a free A800 GPU (allowed: 6 or 7)"
while true; do
    FREE_GPU=$(select_free_gpu) && break
    echo "[$(date -Iseconds)] No free GPU on A800 (allowed: 6 or 7), waiting..."
    sleep 60
done

export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
echo "GPU ${FREE_GPU} is free; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Run exactly one epoch of VoxelPose training.
echo "Launching VoxelPose 1-epoch GPU check on A800 GPU ${FREE_GPU}..."
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${VOXELPOSE_DIR}/run/train_3d.py" --cfg "${RUN_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "[$(date -Iseconds)] VoxelPose GPU check FAILED with exit code ${EXIT_CODE}" >&2
    exit "${EXIT_CODE}"
fi

echo "[$(date -Iseconds)] VoxelPose true-GT v2 GPU check finished successfully."
