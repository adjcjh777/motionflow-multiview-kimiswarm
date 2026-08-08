#!/usr/bin/env bash
# v25 ablation matrix launcher.
#
# Validates and (optionally) runs the v25 ablation matrix defined in
# configs/ablations/v25_ablation_matrix.yaml via scripts/run_ablation_variants.py.
#
# Usage
# -----
#   # Dry-run: print commands without executing (default)
#   bash scripts/run_v25_ablation_matrix.sh
#
#   # Execute all variants sequentially
#   bash scripts/run_v25_ablation_matrix.sh --run
#
#   # Execute with up to N variants in parallel
#   bash scripts/run_v25_ablation_matrix.sh --run --max-workers 2
#
#   # Use a custom Python interpreter
#   PYTHON=/path/to/python bash scripts/run_v25_ablation_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/ablations/v25_ablation_matrix.yaml"
RUNNER="${REPO_ROOT}/scripts/run_ablation_variants.py"
PYTHON=${PYTHON:-python}

# Parse arguments.
RUN=0
MAX_WORKERS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN=1
            shift
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--run] [--max-workers N]" >&2
            exit 1
            ;;
    esac
done

# Validate config and print commands first.
mkdir -p "${REPO_ROOT}/outputs/ablations"
cd "${REPO_ROOT}"

if [[ ${RUN} -eq 0 ]]; then
    echo "=== v25 ablation matrix (dry-run) ==="
    ${PYTHON} "${RUNNER}" --config "${CONFIG}" --dry-run
    echo
    echo "Dry-run complete. Re-run with --run to execute."
else
    echo "=== v25 ablation matrix (running with max-workers=${MAX_WORKERS}) ==="
    ${PYTHON} "${RUNNER}" --config "${CONFIG}" --max-workers "${MAX_WORKERS}"
fi
