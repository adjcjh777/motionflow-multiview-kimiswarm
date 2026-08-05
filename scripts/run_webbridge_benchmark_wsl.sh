#!/usr/bin/env bash
# Run the unified WebBridge benchmark on the best cross-view PP checkpoint.
set -e
cd "$(dirname "$0")/.."

. /tmp/mf_venv/bin/activate

python experiments/run_webbridge_benchmark.py \
    --manifest configs/benchmark_webbridge_mpi_smoke.yaml \
    --out outputs/webbridge_benchmark_mpi_smoke
