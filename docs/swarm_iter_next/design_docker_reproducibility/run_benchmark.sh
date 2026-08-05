#!/usr/bin/env bash
# Run the A800 benchmark inside the MotionFlow-MultiView container.
#
# Usage (inside the container):
#     bash /workspace/motionflow-multiview/docs/swarm_iter_next/design_docker_reproducibility/run_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running A800 benchmark ..."
python3 scripts/benchmark_a800.py \
    --batch_sizes 1 8 16 32 64 \
    --clip_len 13 \
    --j 28 \
    --d 64 \
    --residual_hidden 128 \
    --warmup 20 \
    --iters 200 \
    --out_dir outputs/benchmark_a800

echo "Benchmark complete. See outputs/benchmark_a800/"
