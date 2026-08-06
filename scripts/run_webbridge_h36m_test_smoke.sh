#!/usr/bin/env bash
# Launch a WebBridge benchmark smoke test on H36M S9/S11 (meters).
# This script is a GPU skeleton; do not run while the RTX 4090 is training.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
# shellcheck source=/dev/null
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

# Optional: wait for the running training job to finish before scheduling.
# GPU_ID=${1:-0}
# if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
#     echo "GPU training active; aborting to avoid collision."
#     exit 1
# fi

python experiments/run_webbridge_benchmark.py \
    --manifest configs/benchmark_webbridge_h36m_test_smoke.yaml \
    --out outputs/webbridge_benchmark_h36m_test_smoke/results

python experiments/summarize_webbridge_benchmark.py \
    --json outputs/webbridge_benchmark_h36m_test_smoke/results.json \
    --out docs/swarm_iter_next/webbridge_h36m_test_smoke_results.md
