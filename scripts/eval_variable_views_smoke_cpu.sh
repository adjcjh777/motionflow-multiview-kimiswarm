#!/usr/bin/env bash
# CPU-only smoke test for variable-view inference.
# Does not touch data/checkpoints and therefore will not interfere with GPU jobs.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="tmp/variable_view_smoke"
mkdir -p "$OUT_DIR"

# Use the system python (Anaconda CPU build) because the .venv symlink is
# currently broken on this WSL session.  The smoke test needs no CUDA.
KMP_DUPLICATE_LIB_OK=TRUE \
    python experiments/eval_variable_views.py \
        --n_views 6 \
        --j 17 \
        --clip_len 9 \
        --num_subsets_per_k 10 \
        --seed 42 \
        --output_json "$OUT_DIR/results.json" \
        --output_csv "$OUT_DIR/results.csv"

echo "Smoke results written to $OUT_DIR"
