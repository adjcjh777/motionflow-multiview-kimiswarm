#!/usr/bin/env bash
# Run geometric baselines on the corrected H36M true-GT v2 split.
#
# The manifest points to data/h36m_true_gt_v2/, which uses true mocap 3D GT
# and is physically consistent (direct MJE in the tens of mm, not 0 or 16 m).
#
# Usage (on A800-D with GPU 6/7 free, or locally with CUDA device):
#   bash scripts/run_h36m_true_gt_v2_baselines.sh
#
# The script runs on GPU 6 by default when launched on A800.

set -euo pipefail

CONFIG=${CONFIG:-configs/splits/h36m_true_gt_v2_standard.yaml}
DEVICE=${DEVICE:-cuda}
OUT_DIR=${OUT_DIR:-outputs/h36m_true_gt_v2_baselines}

mkdir -p "$OUT_DIR"

PYTHON=${PYTHON:-python}

echo "Running DLT baseline on $CONFIG ..."
$PYTHON scripts/run_h36m_true_gt_dlt_baseline.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --unweighted \
    --output "$OUT_DIR/dlt_baseline_h36m_true_gt_v2.json"

echo "Running RANSAC/conf-DLT baseline on $CONFIG ..."
$PYTHON scripts/run_h36m_true_gt_ransac_baseline.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --output "$OUT_DIR/ransac_baseline_h36m_true_gt_v2.json"

echo "All v2 baselines complete. Results: $OUT_DIR"
