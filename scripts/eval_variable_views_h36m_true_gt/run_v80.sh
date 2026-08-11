#!/usr/bin/env bash
# Variable-view inference benchmark for v80 (learned view-reliability before
# triangulation) on H36M true-GT (S9/S11).
#
# Expected checkpoint: produced by scripts/run_v80_h36m_true_gt_smoke_local_4090.sh
#   outputs/omniview_fusion_v80_h36m_true_gt_smoke.pth
#   outputs/omniview_fusion_v80_h36m_true_gt_smoke.config.json
#
# This script only runs evaluation; it does not start training.  GPU usage is
# read-only inference, but do not run it while another GPU training job is
# active on the local RTX 4090.
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV=${MF_VENV:-.venv}
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck source=/dev/null
    . "$VENV/bin/activate"
fi

export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

CHECKPOINT="${CHECKPOINT:-outputs/omniview_fusion_v80_h36m_true_gt_smoke.pth}"
CONFIG="${CONFIG:-outputs/omniview_fusion_v80_h36m_true_gt_smoke.config.json}"
OUT_DIR="outputs/eval_variable_views_h36m_true_gt"
mkdir -p "$OUT_DIR"

python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --dataset_manifest scripts/eval_variable_views_h36m_true_gt/h36m_true_gt_val_manifest.txt \
    --clip_len 13 \
    --min_views 2 \
    --max_views 4 \
    --num_subsets_per_k 50 \
    --seed 42 \
    --output_json "$OUT_DIR/v80_results.json" \
    --output_csv "$OUT_DIR/v80_results.csv"
