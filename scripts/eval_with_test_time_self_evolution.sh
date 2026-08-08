#!/usr/bin/env bash
# Evaluate a trained v5 checkpoint with v27 test-time self-evolution enabled.
# Usage: bash scripts/eval_with_test_time_self_evolution.sh <checkpoint> <dataset.npz> [out_json]
set -euo pipefail

CKPT="${1:?Usage: $0 <checkpoint> <dataset.npz> [out_json]}"
DATASET="${2:?Usage: $0 <checkpoint> <dataset.npz> [out_json]}"
OUT_JSON="${3:-$(basename "$CKPT" .pth)_tte.json}"
OUT_CSV="${OUT_JSON%.json}.csv"

# Detect whether this is H36M or MPI from the dataset path.
if [[ "$DATASET" == *h36m* ]]; then
    EVAL_SCRIPT="experiments/eval_omniview_fusion_v5_h36m.py"
elif [[ "$DATASET" == *mpi* ]]; then
    EVAL_SCRIPT="experiments/eval_omniview_fusion_v5_mpiinf3dhp.py"
else
    echo "Cannot infer eval script from dataset path: $DATASET"
    exit 1
fi

python "$EVAL_SCRIPT" \
    --checkpoint "$CKPT" \
    --dataset "$DATASET" \
    --use_test_time_self_evolution_v27 \
    --v27_tte_n_iters 3 \
    --run_robustness \
    --run_variable_views \
    --out_json "$OUT_JSON" \
    --out_csv "$OUT_CSV" \
    > "outputs/$(basename "$OUT_JSON" .json).log" 2>&1

echo "TTE eval -> $OUT_JSON"
