#!/usr/bin/env bash
# Evaluate all iter14 smoke checkpoints and produce a summary table.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"

VAL=data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz
OUT_DIR=outputs/iter14_smoke_eval
mkdir -p "$OUT_DIR"

declare -A CKPTS
CKPTS[reprojection_consistency_pp]=outputs/reprojection_consistency_pp_smoke.pth
CKPTS[dynamic_gate_pp]=outputs/dynamic_view_gate_smoke.pth
CKPTS[graph_skeleton_residual_pp]=outputs/graph_skeleton_residual_pp_smoke.pth
CKPTS[epipolar_pp]=outputs/epipolar_pp_smoke.pth

echo "# Iter14 Smoke Evaluation" > "$OUT_DIR/summary.md"
echo "" >> "$OUT_DIR/summary.md"
echo "| model | checkpoint | mpjpe (mm) | pa_mpjpe (mm) | pck@50 | pck@100 | pck@150 | auc |" >> "$OUT_DIR/summary.md"
echo "|-------|------------|------------|---------------|--------|---------|---------|-----|" >> "$OUT_DIR/summary.md"

for model in reprojection_consistency_pp dynamic_gate_pp graph_skeleton_residual_pp epipolar_pp; do
    ckpt=${CKPTS[$model]}
    if [ ! -f "$ckpt" ]; then
        echo "Skipping $model: $ckpt not found"
        continue
    fi
    echo "Evaluating $model -> $ckpt"
    python -u experiments/eval_full_metrics.py \
        --model "$model" \
        --dataset "$VAL" \
        --checkpoint "$ckpt" \
        --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
        --output_json "$OUT_DIR/${model}.json" \
        > "$OUT_DIR/${model}.log" 2>&1 || true
    if [ -f "$OUT_DIR/${model}.json" ]; then
        python -u - <<PY
import json, sys
p = "$OUT_DIR/${model}.json"
try:
    r = json.load(open(p))
    print(f"{model}|{ckpt}|{r['mpjpe']:.2f}|{r['pa_mpjpe']:.2f}|{r['pck@50mm']:.4f}|{r['pck@100mm']:.4f}|{r['pck@150mm']:.4f}|{r['pck_auc']:.4f}")
except Exception as e:
    print(f"{model}|{ckpt}|parse error: {e}")
PY
    fi
done >> "$OUT_DIR/summary.md"

echo "Summary written to $OUT_DIR/summary.md"
