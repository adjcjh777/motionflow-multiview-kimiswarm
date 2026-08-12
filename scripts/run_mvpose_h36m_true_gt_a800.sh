#!/usr/bin/env bash
# A800 launch script for the MVPose baseline on H36M true-GT.
#
# MVPose (zju3dv/mvpose) is the method from:
#   "Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views"
#   Dong et al., CVPR 2019 / T-PAMI 2021.
#
# This script only *prepares* the baseline. It does not run the upstream
# pipeline, because zju3dv/mvpose is a multi-person Campus/Shelf method and
# a custom H36M adapter is still required.
#
# Usage (manual only; do not launch automatically):
#   nohup bash scripts/run_mvpose_h36m_true_gt_a800.sh > outputs/sota_baselines/mvpose_h36m_true_gt_a800_nohup.log 2>&1 &

set -euo pipefail

# A800 working copy of the repository.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
# Default to GPU 6; override with CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/sota_baselines/mvpose_h36m_a800_config.yaml"

mkdir -p outputs/sota_baselines
mkdir -p tmp/sota_baselines

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a outputs/sota_baselines/mvpose_h36m_true_gt_a800.log
}

log "MVPose H36M true-GT A800 preparation start"
log "Config: ${CONFIG}"
log "GPU: ${CUDA_VISIBLE_DEVICES}"

# 1. Export the H36M true-GT common baseline format if not already present.
INPUT_PKL=$("${PYTHON}" - <<'PY'
import yaml, sys
with open("scripts/sota_baselines/mvpose_h36m_a800_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["input_pkl"])
PY
)

if [[ ! -f "${INPUT_PKL}" ]]; then
    log "Exporting H36M true-GT to common baseline format..."
    "${PYTHON}" "${SCRIPT_DIR}/sota_baselines/common_export_h36m_true_gt.py"
else
    log "Common baseline pickle already exists: ${INPUT_PKL}"
fi

# 2. Clone the public MVPose implementation if not already present.
MVPose_DIR=$("${PYTHON}" - <<'PY'
import yaml
with open("scripts/sota_baselines/mvpose_h36m_a800_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["repo"]["dir"])
PY
)

if [[ ! -d "${MVPose_DIR}/.git" ]]; then
    log "Cloning MVPose upstream repository..."
    mkdir -p "$(dirname "${MVPose_DIR}")"
    git clone --depth 1 --branch master https://github.com/zju3dv/mvpose.git "${MVPose_DIR}"
else
    log "MVPose repo already cloned at ${MVPose_DIR}"
fi

# 3. Convert to MVPose input format.
log "Converting H36M true-GT to MVPose input format..."
"${PYTHON}" "${SCRIPT_DIR}/sota_baselines/convert_to_mvpose_format.py" --config "${CONFIG}"

# 4. Adapter check. The upstream code is for multi-person Campus/Shelf and uses
#    per-frame heatmap/crop dictionaries. A custom adapter must be implemented
#    before inference can proceed.
ADAPTER="${SCRIPT_DIR}/sota_baselines/mvpose_h36m_adapter.py"
if [[ ! -f "${ADAPTER}" ]]; then
    log ""
    log "MVPose data preparation is complete, but the H36M adapter is missing."
    log "Create ${ADAPTER} (or equivalent) to drive zju3dv/mvpose on the"
    log "converted manifest and run inference. No training is required."
    log ""
    log "Next steps:"
    log "  1. Implement the adapter that feeds H36M GT 2D projections and"
    log "     camera parameters into zju3dv/mvpose/src/models/estimate3d.py."
    log "  2. Resolve environment blockers (TF 1.9 / torch 1.0 are incompatible"
    log "     with A800 Ampere; see scripts/sota_baselines/README.md)."
    log "  3. Re-run this script to evaluate on S9/S11."
    exit 0
fi

# 5. Run inference via the adapter once it exists.
log "Running MVPose inference through ${ADAPTER}..."
"${PYTHON}" -u "${ADAPTER}" --config "${CONFIG}"

log "MVPose H36M true-GT A800 run complete."
