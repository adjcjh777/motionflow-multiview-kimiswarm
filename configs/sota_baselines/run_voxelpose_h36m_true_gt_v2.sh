#!/usr/bin/env bash
# A800 launcher for the VoxelPose H36M true-GT v2 baseline.
#
# Usage (from the local WSL repo or directly on a800-D):
#   ssh a800-D 'bash /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/configs/sota_baselines/run_voxelpose_h36m_true_gt_v2.sh'
#
# The script exports the v2 common format, converts to VoxelPose input,
# clones/applies the H36M adapter overlay, waits for a free project GPU
# (GPU 6 or 7 only), and starts training. It will block until a GPU is free
# so it never interferes with running jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PREP_CONFIG="${SCRIPT_DIR}/voxelpose_h36m_true_gt_v2_prep.yaml"
RUN_CONFIG="${SCRIPT_DIR}/voxelpose_h36m_true_gt_v2.yaml"

# Upstream VoxelPose checkout and environment.
VOXELPOSE_DIR="${REPO_ROOT}/models/voxelpose-pytorch"
OVERLAY_DIR="${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay"
CONDA_EXE="${CONDA_EXE:-/mnt/nvme0n1p1/zhangzy/tools/miniconda3/bin/conda}"
VENV_NAME="${VENV_NAME:-voxelpose_py38_pt112}"

PYTHON="${PYTHON:-python}"

LOG_DIR="${REPO_ROOT}/outputs/sota_baselines"
LOG_FILE="${LOG_DIR}/voxelpose_h36m_true_gt_v2_run.log"
mkdir -p "${LOG_DIR}" "${REPO_ROOT}/tmp/sota_baselines"

exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "[$(date -Iseconds)] VoxelPose H36M true-GT v2 A800 prep starting"
echo "repo root: ${REPO_ROOT}"
echo "prep config: ${PREP_CONFIG}"
echo "run config:  ${RUN_CONFIG}"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Export H36M true-GT v2 to the common baseline format (CPU-only)
# ---------------------------------------------------------------------------
INPUT_PKL="${REPO_ROOT}/tmp/sota_baselines/h36m_true_gt_v2_baseline_format.pkl"

if [[ ! -f "${INPUT_PKL}" ]]; then
    echo "[1/5] Exporting H36M true-GT v2 to common baseline format..."
    "${PYTHON}" scripts/sota_baselines/common_export_h36m_true_gt.py \
        --split_yaml configs/splits/h36m_true_gt_v2_standard.yaml \
        --output "${INPUT_PKL}"
else
    echo "[1/5] Common baseline format already exists: ${INPUT_PKL}"
fi

# ---------------------------------------------------------------------------
# 2. Convert common format to VoxelPose-specific input
# ---------------------------------------------------------------------------
echo "[2/5] Converting to VoxelPose input format..."
"${PYTHON}" scripts/sota_baselines/convert_to_voxelpose_format.py \
    --config "${PREP_CONFIG}"

# ---------------------------------------------------------------------------
# 3. Clone VoxelPose upstream if not already present
# ---------------------------------------------------------------------------
if [[ -d "${VOXELPOSE_DIR}/.git" ]] || [[ -f "${VOXELPOSE_DIR}/run/train_3d.py" ]]; then
    echo "[3/5] VoxelPose repo already present."
else
    echo "[3/5] Cloning microsoft/voxelpose-pytorch into ${VOXELPOSE_DIR}..."
    mkdir -p "$(dirname "${VOXELPOSE_DIR}")"
    git clone --depth 1 https://github.com/microsoft/voxelpose-pytorch.git "${VOXELPOSE_DIR}"
fi

# ---------------------------------------------------------------------------
# 4. Apply the H36M adapter overlay
# ---------------------------------------------------------------------------
echo "[4/5] Applying H36M true-GT adapter overlay..."
cp "${OVERLAY_DIR}/h36m_true_gt.py" "${VOXELPOSE_DIR}/lib/dataset/h36m_true_gt.py"
cp "${OVERLAY_DIR}/__init__.py" "${VOXELPOSE_DIR}/lib/dataset/__init__.py"
cp "${OVERLAY_DIR}/patch_voxelpose_function.py" "${VOXELPOSE_DIR}/lib/core/patch_voxelpose_function.py"
cd "${VOXELPOSE_DIR}"
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    lib/core/patch_voxelpose_function.py lib/core/function.py
cd "${REPO_ROOT}"
mkdir -p "${VOXELPOSE_DIR}/output" "${VOXELPOSE_DIR}/log"

# ---------------------------------------------------------------------------
# 5. Wait until either GPU 6 or 7 is free (memory used < 1000 MiB).
#    Project policy: only GPUs 6 and 7 may be used.
# ---------------------------------------------------------------------------
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

echo "[5/5] Waiting for a free A800 GPU (allowed: 6 or 7)"

while true; do
    FREE_GPU=$(select_free_gpu) && break
    echo "[$(date -Iseconds)] No free GPU on A800 (allowed: 6 or 7), waiting..."
    sleep 60
done

export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
echo "GPU ${FREE_GPU} is free; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---------------------------------------------------------------------------
# 6. Launch training
# ---------------------------------------------------------------------------
echo "Launching VoxelPose training on A800 GPU ${FREE_GPU}..."
echo "    config: ${RUN_CONFIG}"
echo "    log:    ${LOG_FILE}"

"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${VOXELPOSE_DIR}/run/train_3d.py" --cfg "${RUN_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date -Iseconds)] VoxelPose H36M true-GT v2 A800 run finished."
