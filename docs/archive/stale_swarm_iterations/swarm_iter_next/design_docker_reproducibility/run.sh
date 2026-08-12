#!/usr/bin/env bash
# Run the MotionFlow-MultiView container interactively.
#
# Usage:
#     bash docs/swarm_iter_next/design_docker_reproducibility/run.sh
#
# The repo root, data/ and outputs/ are bind-mounted into the container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_NAME="${MOTIONFLOW_IMAGE:-motionflow-multiview:latest}"

# Create host directories if they do not exist.
mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/outputs"

echo "Launching container from image: ${IMAGE_NAME}"
docker run --rm -it \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1:-1 \
    -e PYTHONHASHSEED=0 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "${PROJECT_ROOT}:/workspace/motionflow-multiview:rw" \
    -v "${PROJECT_ROOT}/data:/workspace/motionflow-multiview/data:rw" \
    -v "${PROJECT_ROOT}/outputs:/workspace/motionflow-multiview/outputs:rw" \
    -w /workspace/motionflow-multiview \
    "${IMAGE_NAME}" \
    /bin/bash
