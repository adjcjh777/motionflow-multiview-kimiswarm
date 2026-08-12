#!/usr/bin/env bash
# A800 launcher for the VoxelPose H36M true-GT baseline.
#
# This script is intended to be run directly on a800-D. From the local WSL
# repo you can invoke it with:
#
#     ssh a800-D 'bash /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/scripts/run_voxelpose_h36m_true_gt_a800.sh'
#
# It performs CPU-only data preparation, clones the Microsoft VoxelPose repo,
# applies the H36M adapter overlay, and starts training. It exits early if no
# A800 GPU is free.
#
# IMPORTANT: Do not run this while another training/evaluation job is active.

set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
VOXELPOSE_DIR="${REPO_ROOT}/models/voxelpose-pytorch"
OVERLAY_DIR="${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay"

# VoxelPose requires the dedicated Python 3.8 + PyTorch 1.12.1 environment
# created by scripts/sota_baselines/setup_voxelpose_env_a800.sh.
VENV_NAME="${VENV_NAME:-voxelpose_py38_pt112}"

# Locate the same conda executable used by the setup script.
CONDA_EXE="${CONDA_EXE:-/mnt/nvme0n1p1/zhangzy/tools/miniconda3/bin/conda}"
if [[ ! -x "${CONDA_EXE}" ]]; then
    echo "ERROR: conda not found at ${CONDA_EXE}. Set CONDA_EXE to the full path." >&2
    exit 1
fi

# Verify the environment exists.
if ! "${CONDA_EXE}" env list | grep -qE "^\s*${VENV_NAME}\s+"; then
    echo "ERROR: conda env '${VENV_NAME}' not found. Run setup_voxelpose_env_a800.sh first." >&2
    exit 1
fi

CONFIG="${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800.yaml"

LOG_DIR="${REPO_ROOT}/outputs/sota_baselines"
LOG_FILE="${LOG_DIR}/voxelpose_h36m_true_gt_a800_run.log"

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "[$(date -Iseconds)] VoxelPose H36M true-GT A800 prep starting"
echo "repo root: ${REPO_ROOT}"
echo "voxelpose: ${VOXELPOSE_DIR}"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Export H36M true-GT to the common baseline format (CPU-only)
# ---------------------------------------------------------------------------
if [[ ! -f "tmp/sota_baselines/h36m_true_gt_baseline_format.pkl" ]]; then
    echo "[1/6] Exporting H36M true-GT to common baseline format..."
    "${CONDA_EXE}" run -n "${VENV_NAME}" python \
        "scripts/sota_baselines/common_export_h36m_true_gt.py" \
        --split_yaml "configs/splits/h36m_true_gt_standard.yaml" \
        --output "tmp/sota_baselines/h36m_true_gt_baseline_format.pkl"
else
    echo "[1/6] Common baseline format already exists."
fi

# ---------------------------------------------------------------------------
# 2. Convert common format to VoxelPose-specific pickles
# ---------------------------------------------------------------------------
echo "[2/6] Converting to VoxelPose input format..."
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "scripts/sota_baselines/convert_to_voxelpose_format.py" \
    --config "scripts/sota_baselines/voxelpose_h36m_config.yaml"

# ---------------------------------------------------------------------------
# 3. Clone VoxelPose upstream if not already present
# ---------------------------------------------------------------------------
if [[ -d "${VOXELPOSE_DIR}/.git" ]] || [[ -f "${VOXELPOSE_DIR}/run/train_3d.py" ]]; then
    echo "[3/6] VoxelPose repo already present."
else
    echo "[3/6] Cloning microsoft/voxelpose-pytorch into ${VOXELPOSE_DIR}..."
    mkdir -p "$(dirname "${VOXELPOSE_DIR}")"
    git clone --depth 1 https://github.com/microsoft/voxelpose-pytorch.git "${VOXELPOSE_DIR}"
fi

# ---------------------------------------------------------------------------
# 4. Apply the H36M adapter overlay
# ---------------------------------------------------------------------------
echo "[4/6] Applying H36M true-GT adapter overlay..."
cp "${OVERLAY_DIR}/h36m_true_gt.py" "${VOXELPOSE_DIR}/lib/dataset/h36m_true_gt.py"
cp "${OVERLAY_DIR}/__init__.py" "${VOXELPOSE_DIR}/lib/dataset/__init__.py"
cp "${OVERLAY_DIR}/patch_voxelpose_function.py" "${VOXELPOSE_DIR}/lib/core/patch_voxelpose_function.py"
cd "${VOXELPOSE_DIR}"
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "lib/core/patch_voxelpose_function.py" "lib/core/function.py"

# VoxelPose's create_logger uses mkdir() without parents=True, so the parent
# output/ and log/ directories must exist before training starts.
mkdir -p "${VOXELPOSE_DIR}/output" "${VOXELPOSE_DIR}/log"

# ---------------------------------------------------------------------------
# 5. GPU free-check (training only; prep can run without a GPU)
# Project policy: only GPUs 6 and 7 may be used.
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

FREE_GPU=$(select_free_gpu) || {
    echo "ERROR: No free GPU on A800 (allowed: 6 or 7). Aborting VoxelPose launch." >&2
    exit 1
}
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
echo "[5/6] GPU ${FREE_GPU} is free; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---------------------------------------------------------------------------
# 6. Launch training
# ---------------------------------------------------------------------------
echo "[6/6] Launching VoxelPose training on A800 GPU ${FREE_GPU}..."
echo "    config: ${CONFIG}"
echo "    log:    ${LOG_FILE}"

# Run in the foreground so the caller sees progress. Wrap with nohup manually
# if you want it to survive SSH disconnect.
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${VOXELPOSE_DIR}/run/train_3d.py" --cfg "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date -Iseconds)] VoxelPose H36M true-GT A800 run finished."
