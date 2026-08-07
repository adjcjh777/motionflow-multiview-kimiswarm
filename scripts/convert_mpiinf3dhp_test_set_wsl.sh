#!/usr/bin/env bash
# Convert MPI-INF-3DHP test set (TS1-TS6) to canonical multi-view .npz.
#
# Usage:
#   bash scripts/convert_mpiinf3dhp_test_set_wsl.sh
#
# Optional environment variables:
#   TEST_ROOT  - folder containing TS1..TS6
#   CALIB      - reference camera.calibration file
#   OUT_DIR    - destination directory for .npz files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEST_ROOT="${TEST_ROOT:-${PROJECT_ROOT}/data/webbridge/mpi_inf_3dhp/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set}"
CALIB="${CALIB:-${PROJECT_ROOT}/data/webbridge/mpi_inf_3dhp/raw/S1/Seq1/camera.calibration}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/webbridge/mpi_inf_3dhp/test_set}"

echo "Converting MPI-INF-3DHP test set..."
echo "  test_root : ${TEST_ROOT}"
echo "  calib     : ${CALIB}"
echo "  out_dir   : ${OUT_DIR}"

mkdir -p "${OUT_DIR}"

python "${PROJECT_ROOT}/experiments/prototypes/swarm_iter18/convert_mpiinf3dhp_test_set.py" \
    --test_root "${TEST_ROOT}" \
    --calib "${CALIB}" \
    --out_dir "${OUT_DIR}" \
    --camera_index 0

echo "MPI-INF-3DHP test-set conversion complete. Output: ${OUT_DIR}"
