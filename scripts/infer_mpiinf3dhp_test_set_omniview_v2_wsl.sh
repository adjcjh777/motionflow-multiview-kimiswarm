#!/usr/bin/env bash
# Run OmniMultiViewFusionV2 inference on the MPI-INF-3DHP test set (WSL).
#
# Usage:
#   bash scripts/infer_mpiinf3dhp_test_set_omniview_v2_wsl.sh [checkpoint]
#
# Environment variables:
#   CHECKPOINT - path to trained OmniMultiViewFusionV2 checkpoint
#   OUT_NPZ    - output predictions path
#   TEST_DIR   - directory containing TS{i}_v14_multiview.npz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${1:-${CHECKPOINT:-outputs/omniview_fusion_v2_mpiinf3dhp.pth}}"
OUT_NPZ="${OUT_NPZ:-${PROJECT_ROOT}/outputs/omniview_fusion_v2_test_set_predictions.npz}"
TEST_DIR="${TEST_DIR:-${PROJECT_ROOT}/data/webbridge/mpi_inf_3dhp/test_set}"

echo "Running OmniMultiViewFusionV2 test-set inference..."
echo "  checkpoint : ${CHECKPOINT}"
echo "  test_dir   : ${TEST_DIR}"
echo "  out_npz    : ${OUT_NPZ}"

python "${PROJECT_ROOT}/experiments/infer_mpiinf3dhp_test_set_omniview_v2.py" \
    --checkpoint "${CHECKPOINT}" \
    --test_set_dir "${TEST_DIR}" \
    --out_npz "${OUT_NPZ}"

echo "Test-set inference complete. Output: ${OUT_NPZ}"
