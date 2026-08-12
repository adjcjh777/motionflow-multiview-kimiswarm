#!/usr/bin/env bash
# Prepare and run VoxelPose on the corrected H36M true-GT protocol.
#
# This script is meant to be launched manually when the local RTX 4090 is free.
# It will exit early if the GPU is busy or if the repo is on the read-only
# A800-D mount.
#
# IMPORTANT: VoxelPose training is run on A800 using the dedicated conda
# environment (see scripts/run_voxelpose_h36m_true_gt_a800.sh). This local
# script only prepares the converted data; it does not run training because
# the upstream code requires PyTorch 1.x / Python 3.8.
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

# 2. Check / clone VoxelPose repository.
#    For actual training, use scripts/run_voxelpose_h36m_true_gt_a800.sh on A800.
if [[ -d "${VOXELPOSE_DIR}/.git" ]] || [[ -f "${VOXELPOSE_DIR}/run/train_3d.py" ]]; then
    echo "VoxelPose repo already present at ${VOXELPOSE_DIR}."
    VOXELPOSE_READY=true
elif [[ ! -d "${VOXELPOSE_DIR}" ]]; then
    echo "Cloning Microsoft VoxelPose repo into ${VOXELPOSE_DIR}..."
    mkdir -p "$(dirname "${VOXELPOSE_DIR}")"
    git clone --depth 1 "${REPO_URL}" "${VOXELPOSE_DIR}"
    VOXELPOSE_READY=true
else
    echo "WARN: ${VOXELPOSE_DIR} exists but does not look like the VoxelPose repo." >&2
    echo "      Please remove it or clone microsoft/voxelpose-pytorch there." >&2
    VOXELPOSE_READY=false
fi

# 3. Convert common baseline format to VoxelPose input format.
echo "Converting to VoxelPose input format..."
"${PYTHON}" "${SCRIPT_DIR}/convert_to_voxelpose_format.py" \
    --config "${CONFIG}"

# 4. Run VoxelPose training/evaluation only if the upstream repo is present.
#    The exact command line depends on the upstream revision.
if [[ "${VOXELPOSE_READY}" != "true" ]]; then
    echo ""
    echo "VoxelPose data prepared, but upstream repository is missing." >&2
    echo "To train/eval, run on A800:" >&2
    echo "  bash scripts/run_voxelpose_h36m_true_gt_a800.sh" >&2
    exit 0
fi

# Local training is intentionally not supported because the upstream code needs
# PyTorch 1.x / Python 3.8. Point the user to the A800 launcher.
echo ""
echo "VoxelPose data prepared. Training must be run on A800:" >&2
echo "  bash scripts/run_voxelpose_h36m_true_gt_a800.sh" >&2
exit 0

