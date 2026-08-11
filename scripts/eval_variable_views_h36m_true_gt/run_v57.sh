#!/usr/bin/env bash
# Variable-view inference benchmark for v57 (domain-conditional PSC) on H36M true-GT (S9/S11).
#
# Expected checkpoint: a v57 H36M true-GT training run, e.g.
#   outputs/omniview_fusion_v57_h36m_true_gt_medium.pth
#   outputs/omniview_fusion_v57_h36m_true_gt_medium.config.json
#
# This checkpoint does not exist yet; generate it first by training v57 on
# configs/splits/h36m_true_gt_standard.yaml.
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

CHECKPOINT="${CHECKPOINT:-outputs/omniview_fusion_v57_h36m_true_gt_medium.pth}"
CONFIG="${CONFIG:-outputs/omniview_fusion_v57_h36m_true_gt_medium.config.json}"
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
    --output_json "$OUT_DIR/v57_results.json" \
    --output_csv "$OUT_DIR/v57_results.csv"
