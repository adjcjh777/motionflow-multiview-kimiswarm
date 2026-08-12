#!/usr/bin/env bash
# Regenerate all non-circular H36M true-GT canonical .npz files.
#
# These .npz use the official mocap 3D GT, not DLT triangulation of the
# input 2D, and are written in meters to data/h36m_true_gt_v2/.
#
# Usage:
#   bash scripts/convert_all_h36m_true_gt_v2.sh
#
# The script is CPU-bound and can run on any available host (WSL or A800).
# GPU 6/7 are left untouched.

set -euo pipefail

PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-data/h36m_true_gt_v2}

mkdir -p "$OUT_DIR"

# Train subjects: S1, S5, S6, S7, S8; all actions 2-16.
echo "Generating train split .npz ..."
for s in 1 5 6 7 8; do
    echo "  Subject $s (train)"
    $PYTHON scripts/convert_h36m_true_gt_v2.py \
        --subject "$s" \
        --actions $(seq 2 16) \
        --split train \
        --out_dir "$OUT_DIR"
done

# Test subjects: S9, S11; all actions 2-16.
echo "Generating test split .npz ..."
for s in 9 11; do
    echo "  Subject $s (test)"
    $PYTHON scripts/convert_h36m_true_gt_v2.py \
        --subject "$s" \
        --actions $(seq 2 16) \
        --split test \
        --out_dir "$OUT_DIR"
done

echo "All H36M true-GT v2 .npz files written to $OUT_DIR"
echo "Run the following to audit a few files:"
echo "  python scripts/diagnose_circular_labels.py $OUT_DIR/s_01_acts_*.npz"
