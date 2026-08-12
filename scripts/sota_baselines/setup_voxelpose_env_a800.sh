#!/usr/bin/env bash
# Build a VoxelPose-compatible conda environment on A800.
#
# Target stack:
#   - Python 3.8
#   - PyTorch 1.12.1 + torchvision 0.13.1 + torchaudio 0.12.1
#   - CUDA 11.6 (cudatoolkit from conda)
#
# This script also clones the Microsoft VoxelPose repo and applies the H36M
# true-GT adapter overlay used by scripts/run_voxelpose_h36m_true_gt_a800.sh.
#
# Usage (run on a800-D from the repo root):
#   bash scripts/sota_baselines/setup_voxelpose_env_a800.sh
#
# Optional environment overrides:
#   VENV_NAME=voxelpose_py38_pt112
#   PYTORCH_VERSION=1.12.1
#   TORCHVISION_VERSION=0.13.1
#   CUDA_VERSION=11.6
#   CONDA_EXE=/path/to/conda

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
: "${VENV_NAME:=voxelpose_py38_pt112}"
: "${PYTHON_VERSION:=3.8}"
: "${PYTORCH_VERSION:=1.12.1}"
: "${TORCHVISION_VERSION:=0.13.1}"
: "${TORCHAUDIO_VERSION:=0.12.1}"
: "${CUDA_VERSION:=11.6}"

# ---------------------------------------------------------------------------
# Locate conda/mamba
# ---------------------------------------------------------------------------
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA="${CONDA_EXE}"
else
    for cand in \
        /mnt/nvme0n1p1/zhangzy/tools/miniconda3/bin/conda \
        /opt/anaconda3/bin/conda \
        /home/zhangyh/miniconda3/bin/conda \
        /home/zhangyh/anaconda3/bin/conda \
        /opt/conda/bin/conda \
        /root/miniconda3/bin/conda \
        conda
    do
        if [[ -x "$(command -v "${cand}" 2>/dev/null || true)" ]] || [[ -x "${cand}" ]]; then
            CONDA="${cand}"
            break
        fi
    done
fi

if [[ -z "${CONDA:-}" ]] || [[ ! -x "${CONDA}" ]]; then
    echo "ERROR: conda executable not found. Set CONDA_EXE to the full path." >&2
    exit 1
fi

echo "==> Using conda: ${CONDA}"

# Make conda available in this non-interactive shell
CONDA_BASE="$(${CONDA} info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || \
    eval "$(${CONDA} shell.bash hook)"

# ---------------------------------------------------------------------------
# Create the conda environment
# ---------------------------------------------------------------------------
if ! ${CONDA} env list | grep -qE "^\s*${VENV_NAME}\s+"; then
    echo "==> Creating conda environment '${VENV_NAME}' with Python ${PYTHON_VERSION} ..."
    ${CONDA} create -y -n "${VENV_NAME}" python="${PYTHON_VERSION}"
else
    echo "==> Conda environment '${VENV_NAME}' already exists."
fi

echo "==> Activating '${VENV_NAME}' ..."
# Temporarily disable nounset: some MKL activation scripts reference
# unbound environment variables (e.g., MKL_INTERFACE_LAYER).
set +u
conda activate "${VENV_NAME}"
set -u

# Verify we are inside the right environment
if [[ "${CONDA_DEFAULT_ENV:-}" != "${VENV_NAME}" ]]; then
    echo "ERROR: Failed to activate environment ${VENV_NAME}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Install PyTorch + CUDA toolkit via conda
# ---------------------------------------------------------------------------
# The A800 driver (580.173.02 / CUDA 13.0) is backward-compatible with CUDA 11.x.
# We deliberately avoid torch==1.4.0 because it is CUDA-10.1 era and will not
# run on Ampere GPUs.
echo "==> Installing PyTorch ${PYTORCH_VERSION} + torchvision ${TORCHVISION_VERSION} + CUDA ${CUDA_VERSION} ..."

# Map a short CUDA version to a cudatoolkit package name.
case "${CUDA_VERSION}" in
    11.1) CT="cudatoolkit=11.1" ;;
    11.3) CT="cudatoolkit=11.3" ;;
    11.6) CT="cudatoolkit=11.6" ;;
    11.7) CT="cudatoolkit=11.7" ;;
    11.8) CT="cudatoolkit=11.8" ;;
    *) echo "ERROR: CUDA ${CUDA_VERSION} is not pre-configured in this script. Add it or use 11.1/11.3/11.6/11.7/11.8." >&2; exit 1 ;;
esac

${CONDA} install -y \
    pytorch=="${PYTORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    "${CT}" \
    -c pytorch -c nvidia

# ---------------------------------------------------------------------------
# Install VoxelPose Python dependencies (modern versions for Python 3.8)
# ---------------------------------------------------------------------------
# The upstream requirements.txt pins versions that are incompatible with
# Python 3.8 and Ampere GPUs (torch==1.4.0, numpy==1.16.2, etc.). We keep
# the same package set but use current, compatible versions.
echo "==> Installing VoxelPose Python dependencies ..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    "numpy<2,>=1.20" \
    "scipy>=1.7,<1.12" \
    "matplotlib>=3.3,<3.7" \
    "opencv-python-headless>=4.5,<5.0" \
    "pillow>=8.0,<11" \
    "tqdm>=4.60" \
    "easydict" \
    "json-tricks>=3.15.5" \
    "prettytable>=3.0" \
    "tensorboardX>=2.6" \
    "PyYAML>=5.4,<7"

# ---------------------------------------------------------------------------
# Clone Microsoft VoxelPose repo and apply the H36M true-GT adapter overlay
# ---------------------------------------------------------------------------
VOXELPOSE_DIR="${REPO_ROOT}/models/voxelpose-pytorch"
OVERLAY_DIR="${REPO_ROOT}/scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay"

if [[ -d "${VOXELPOSE_DIR}/.git" ]] || [[ -f "${VOXELPOSE_DIR}/run/train_3d.py" ]]; then
    echo "==> VoxelPose repo already present at ${VOXELPOSE_DIR}"
else
    echo "==> Cloning microsoft/voxelpose-pytorch into ${VOXELPOSE_DIR} ..."
    mkdir -p "$(dirname "${VOXELPOSE_DIR}")"
    git clone --depth 1 https://github.com/microsoft/voxelpose-pytorch.git "${VOXELPOSE_DIR}"
fi

echo "==> Applying H36M true-GT adapter overlay ..."
cp "${OVERLAY_DIR}/h36m_true_gt.py" "${VOXELPOSE_DIR}/lib/dataset/h36m_true_gt.py"
cp "${OVERLAY_DIR}/__init__.py" "${VOXELPOSE_DIR}/lib/dataset/__init__.py"
cp "${OVERLAY_DIR}/patch_voxelpose_function.py" "${VOXELPOSE_DIR}/lib/core/patch_voxelpose_function.py"
python "${VOXELPOSE_DIR}/lib/core/patch_voxelpose_function.py" "${VOXELPOSE_DIR}/lib/core/function.py"

# ---------------------------------------------------------------------------
# Sanity check: can we import torch and the VoxelPose modules?
# ---------------------------------------------------------------------------
echo "==> Running quick Python sanity checks ..."
python - <<'PY'
import sys
import torch
import torchvision
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"torchvision: {torchvision.__version__}")
PY

# ---------------------------------------------------------------------------
# Summary / next steps
# ---------------------------------------------------------------------------
cat <<EOF

===========================================================================
VoxelPose A800 environment setup complete.

Environment name : ${VENV_NAME}
Conda executable : ${CONDA}
PyTorch version  : ${PYTORCH_VERSION}
CUDA toolkit     : ${CUDA_VERSION}
VoxelPose repo   : ${VOXELPOSE_DIR}

Activate the environment on A800:
    conda activate ${VENV_NAME}

Run the baseline (only when an A800 GPU is free):
    bash scripts/run_voxelpose_h36m_true_gt_a800.sh

Porting notes:
1. Original upstream pins torch==1.4.0/cu10.1, which cannot run on Ampere.
2. Python 3.8 + PyTorch 1.12.1 + cudatoolkit 11.6 keeps VoxelPose APIs
   (grid_sample align_corners, DataParallel, etc.) unchanged.
3. numpy, scipy, matplotlib, and opencv versions were lifted to current
   Python-3.8-compatible releases.
4. If you need PyTorch 1.8 instead, override before running:
     PYTORCH_VERSION=1.8.2 TORCHVISION_VERSION=0.9.1 TORCHAUDIO_VERSION=0.8.1 CUDA_VERSION=11.1 bash scripts/sota_baselines/setup_voxelpose_env_a800.sh
5. Training is intentionally NOT started by this script.
===========================================================================
EOF
