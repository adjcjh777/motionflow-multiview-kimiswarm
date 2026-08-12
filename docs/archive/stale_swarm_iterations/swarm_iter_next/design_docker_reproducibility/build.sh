#!/usr/bin/env bash
# Build the MotionFlow-MultiView reproducibility Docker image.
#
# Usage:
#     bash docs/swarm_iter_next/design_docker_reproducibility/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_NAME="${MOTIONFLOW_IMAGE:-motionflow-multiview:latest}"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${PROJECT_ROOT}"

echo "Built ${IMAGE_NAME}"
