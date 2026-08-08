#!/usr/bin/env bash
# Smoke-run the v25 failure-mode analysis on a synthetic validation set.
#
# Usage:
#   bash scripts/run_v25_failure_analysis.sh
#   bash scripts/run_v25_failure_analysis.sh /path/to/checkpoint.pth
#
# The script runs entirely on CPU by default, so it is safe to run locally
# while A800-D GPUs are busy.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${1:-}"
OUT_DIR="outputs/v25_failure_analysis"

mkdir -p "$OUT_DIR"

echo "Running v25 failure-mode analysis (synthetic smoke test)..."

if [ -n "$CHECKPOINT" ]; then
    "$PYTHON" scripts/analyze_v25_failures.py \
        --checkpoint "$CHECKPOINT" \
        --synthetic \
        --out_dir "$OUT_DIR" \
        --device cpu \
        --batch_size 16
else
    "$PYTHON" scripts/analyze_v25_failures.py \
        --synthetic \
        --out_dir "$OUT_DIR" \
        --device cpu \
        --batch_size 16
fi

echo "Done. See $OUT_DIR/v25_failure_analysis_report.md"
