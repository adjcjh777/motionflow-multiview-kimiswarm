#!/usr/bin/env bash
# Prepare and run VoxelPose on the corrected H36M true-GT protocol.
#
# This script is meant to be launched manually when the local RTX 4090 is free.
# It will exit early if the GPU is busy or if the repo is on the read-only
# A800-D mount.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/voxelpose_h36m_config.yaml"

PYTHON=${PYTHON:-python}

cd "${REPO_ROOT}"

# Safety guard: never run on A800-D / read-only mount.
HOSTNAME=$(hostname 2>/dev/null || uname -n)
if [[ "${HOSTNAME}" == "a800-D"* ]] || [[ -d /mnt/nvme0n1/zhangzy/projects ]]; then
    echo "ERROR: This script must not run on A800-D (read-only)." >&2
    exit 1
fi

# Safety guard: do not start if the GPU is busy.
if ! bash "${SCRIPT_DIR}/check_gpu_free.sh"; then
    echo "GPU is not free. Aborting." >&2
    exit 1
fi

# Load helper values from the YAML config.
VOXELPOSE_DIR=$(python - <<'PY'
import yaml, sys
with open("scripts/sota_baselines/voxelpose_h36m_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["repo"]["dir"])
PY
)
REPO_URL=$(python - <<'PY'
import yaml
with open("scripts/sota_baselines/voxelpose_h36m_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["repo"]["url"])
PY
)
INPUT_PKL=$(python - <<'PY'
import yaml
with open("scripts/sota_baselines/voxelpose_h36m_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["input_pkl"])
PY
)

# 1. Export H36M true-GT to the common baseline format if needed.
if [[ ! -f "${INPUT_PKL}" ]]; then
    echo "Exporting H36M true-GT to common baseline format..."
    "${PYTHON}" "${SCRIPT_DIR}/common_export_h36m_true_gt.py"
fi

# 2. Clone VoxelPose repository if not present.
if [[ ! -d "${VOXELPOSE_DIR}/.git" ]]; then
    echo "Cloning VoxelPose from ${REPO_URL}..."
    mkdir -p "$(dirname "${VOXELPOSE_DIR}")"
    git clone "${REPO_URL}" "${VOXELPOSE_DIR}"
else
    echo "VoxelPose repo already cloned at ${VOXELPOSE_DIR}."
fi

# 3. Convert common baseline format to VoxelPose input format.
echo "Converting to VoxelPose input format..."
"${PYTHON}" "${SCRIPT_DIR}/convert_to_voxelpose_format.py" \
    --config "${CONFIG}"

# 4. Run VoxelPose training (or evaluation if checkpoint already exists).
#    This is the GPU-backed step and is gated by check_gpu_free.sh above.
CHECKPOINT_DIR="${VOXELPOSE_DIR}/output/h36m_true_gt/final_state.pth"
if [[ -f "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint exists; running evaluation."
    "${PYTHON}" "${VOXELPOSE_DIR}/run/test.py" \
        --cfg "${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_run_config.yaml" \
        --model "${CHECKPOINT_DIR}"
else
    echo "Starting VoxelPose training..."
    "${PYTHON}" "${VOXELPOSE_DIR}/run/train.py" \
        --cfg "${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_run_config.yaml"
fi

echo "VoxelPose prep/run complete. See outputs/sota_baselines/ for logs/metrics."
