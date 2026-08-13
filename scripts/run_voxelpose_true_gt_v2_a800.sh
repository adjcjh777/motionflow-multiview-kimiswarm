#!/usr/bin/env bash
# A800 launcher/evaluation script for the VoxelPose H36M true-GT v2 baseline.
#
# Usage (run directly on a800-D or via SSH from WSL):
#   ssh a800-D 'bash /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/scripts/run_voxelpose_true_gt_v2_a800.sh'
#
# The script:
#   1. Exports the corrected H36M true-GT v2 split to the common baseline pickle.
#   2. Converts the common format to VoxelPose-specific input pickles.
#   3. Ensures the upstream VoxelPose repo is present and applies the H36M adapter.
#   4. Waits for a free project GPU (GPU 6 or 7 only; never touches 0-5).
#   5. Runs VoxelPose training + validation.
#   6. Extracts the best MPJPE and writes
#      outputs/sota_baselines/voxelpose_true_gt_v2.json.
#
# Environment variables:
#   CONDA_EXE    - path to conda executable (default: A800 miniconda3)
#   VENV_NAME    - conda env with Python 3.8 + PyTorch 1.12.1 (default: voxelpose_py38_pt112)
#   SKIP_TRAIN   - if 1, skip training and only run final validation on model_best.pth.tar
#   SKIP_PREP    - if 1, skip data export/conversion (use existing pickles)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PREP_CONFIG="${REPO_ROOT}/configs/sota_baselines/voxelpose_h36m_true_gt_v2_prep.yaml"
RUN_CONFIG="${REPO_ROOT}/configs/sota_baselines/voxelpose_h36m_true_gt_v2.yaml"

VOXELPOSE_DIR="${REPO_ROOT}/models/voxelpose-pytorch"
OVERLAY_DIR="${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay"
CONDA_EXE="${CONDA_EXE:-/mnt/nvme0n1p1/zhangzy/tools/miniconda3/bin/conda}"
VENV_NAME="${VENV_NAME:-voxelpose_py38_pt112}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PREP="${SKIP_PREP:-0}"

OUTPUT_DIR="${REPO_ROOT}/outputs/sota_baselines"
LOG_DIR="${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/voxelpose_true_gt_v2_a800_run.log"
RESULT_JSON="${OUTPUT_DIR}/voxelpose_true_gt_v2.json"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/tmp/sota_baselines"

exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "[$(date -Iseconds)] VoxelPose H36M true-GT v2 A800 run starting"
echo "repo root: ${REPO_ROOT}"
echo "prep config: ${PREP_CONFIG}"
echo "run config:  ${RUN_CONFIG}"
echo "result JSON: ${RESULT_JSON}"

cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# 0. Validate environment
# -----------------------------------------------------------------------------
if [[ ! -x "${CONDA_EXE}" ]]; then
    echo "ERROR: conda not found at ${CONDA_EXE}. Set CONDA_EXE." >&2
    exit 1
fi

if ! "${CONDA_EXE}" env list | grep -qE "^\s*${VENV_NAME}\s+"; then
    echo "ERROR: conda env '${VENV_NAME}' not found. Run setup_voxelpose_env_a800.sh first." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. Export H36M true-GT v2 to the common baseline format (CPU-only)
# -----------------------------------------------------------------------------
INPUT_PKL="${REPO_ROOT}/tmp/sota_baselines/h36m_true_gt_v2_baseline_format.pkl"

if [[ "${SKIP_PREP}" -eq 0 ]]; then
    echo "[1/5] Exporting H36M true-GT v2 to common baseline format..."
    "${CONDA_EXE}" run -n "${VENV_NAME}" python \
        "${REPO_ROOT}/scripts/sota_baselines/common_export_h36m_true_gt.py" \
        --split_yaml "${REPO_ROOT}/configs/splits/h36m_true_gt_v2_standard.yaml" \
        --output "${INPUT_PKL}"
else
    echo "[1/5] SKIP_PREP set; reusing ${INPUT_PKL}"
fi

# -----------------------------------------------------------------------------
# 2. Convert common format to VoxelPose-specific input
# -----------------------------------------------------------------------------
echo "[2/5] Converting to VoxelPose input format..."
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${REPO_ROOT}/scripts/sota_baselines/convert_to_voxelpose_format.py" \
    --config "${PREP_CONFIG}"

# -----------------------------------------------------------------------------
# 3. Ensure upstream VoxelPose is present and apply the H36M adapter overlay
# -----------------------------------------------------------------------------
echo "[3/5] Ensuring VoxelPose repo is present and overlay applied..."
if [[ ! -d "${VOXELPOSE_DIR}/.git" ]] && [[ ! -f "${VOXELPOSE_DIR}/run/train_3d.py" ]]; then
    echo "Cloning microsoft/voxelpose-pytorch into ${VOXELPOSE_DIR}..."
    mkdir -p "$(dirname "${VOXELPOSE_DIR}")"
    git clone --depth 1 https://github.com/microsoft/voxelpose-pytorch.git "${VOXELPOSE_DIR}"
fi

echo "Applying H36M true-GT v2 adapter overlay..."
cp "${OVERLAY_DIR}/h36m_true_gt.py" "${VOXELPOSE_DIR}/lib/dataset/h36m_true_gt.py"
cp "${OVERLAY_DIR}/__init__.py" "${VOXELPOSE_DIR}/lib/dataset/__init__.py"
cp "${OVERLAY_DIR}/patch_voxelpose_function.py" "${VOXELPOSE_DIR}/lib/core/patch_voxelpose_function.py"

"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${VOXELPOSE_DIR}/lib/core/patch_voxelpose_function.py" \
    "${VOXELPOSE_DIR}/lib/core/function.py"

mkdir -p "${VOXELPOSE_DIR}/output" "${VOXELPOSE_DIR}/log"

# -----------------------------------------------------------------------------
# 4. Wait until a project GPU (6 or 7) is free
# -----------------------------------------------------------------------------
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

echo "[4/5] Waiting for a free A800 GPU (allowed: 6 or 7)"

while true; do
    FREE_GPU=$(select_free_gpu) && break
    echo "[$(date -Iseconds)] No free GPU on A800 (allowed: 6 or 7), waiting..."
    sleep 60
done

export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
echo "GPU ${FREE_GPU} is free; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# -----------------------------------------------------------------------------
# 5. Train VoxelPose on H36M true-GT v2
# -----------------------------------------------------------------------------
if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
    echo "[5/5] Launching VoxelPose training on A800 GPU ${FREE_GPU}..."
    echo "    config: ${RUN_CONFIG}"
    echo "    log:    ${LOG_FILE}"

    "${CONDA_EXE}" run -n "${VENV_NAME}" python \
        "${VOXELPOSE_DIR}/run/train_3d.py" --cfg "${RUN_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

    echo "[$(date -Iseconds)] VoxelPose training finished."
else
    echo "[5/5] SKIP_TRAIN set; skipping training."
fi

# -----------------------------------------------------------------------------
# 6. Final validation on the best checkpoint and write JSON result
# -----------------------------------------------------------------------------
echo "Running final validation on model_best.pth.tar..."
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${VOXELPOSE_DIR}/run/validate_3d.py" --cfg "${RUN_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

MODEL_BEST="${VOXELPOSE_DIR}/output/h36m_true_gt_v2_a800/h36m_true_gt_v2/multi_person_posenet/voxelpose_h36m_true_gt_v2/model_best.pth.tar"

echo "Extracting MPJPE from ${LOG_FILE} and writing result JSON..."
"${CONDA_EXE}" run -n "${VENV_NAME}" python \
    "${REPO_ROOT}/scripts/sota_baselines/extract_voxelpose_mpjpe.py" \
    --log "${LOG_FILE}" \
    --run-config "${RUN_CONFIG}" \
    --checkpoint "${MODEL_BEST}" \
    --output "${RESULT_JSON}"

echo "[$(date -Iseconds)] VoxelPose H36M true-GT v2 A800 run finished."
echo "Result JSON: ${RESULT_JSON}"
