#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

process_one() {
    local input_json=$1
    local suffix=$2
    local title=$3

    [ -f "$input_json" ] || return 0

    echo "[postprocess] processing $input_json"
    local output_png="docs/figures/variable_views_${suffix}.png"
    local output_md="docs/results_variable_views_${suffix}.md"

    python experiments/plot_variable_views.py \
        --input "$input_json" \
        --output "$output_png" \
        --title "$title"

    python - <<PY
import json
from pathlib import Path
p = Path('$input_json')
out = Path('$output_md')
data = json.load(open(p))
lines = ['# Variable-View MPJPE@k Results ($suffix)\n\n']
lines.append('| k | mean_mm | std_mm | subsets |\n')
lines.append('|---|---|---|---|\n')
for k in sorted(data.keys(), key=int):
    v = data[k]
    lines.append(f"| {k} | {v['mean_mm']:.4f} | {v['std_mm']:.4f} | {v['n_subsets']} |\n")
out.write_text(''.join(lines))
PY

    echo "[postprocess] wrote $output_png and $output_md"
}

# Wait for any currently running eval_variable_views to finish
echo "[postprocess] waiting for any variable-view eval to finish..."
while pgrep -f "experiments/eval_variable_views.py" > /dev/null; do
    sleep 10
done

process_one "outputs/variable_views_crossview_residual_quick.json" "quick" "MPJPE@k cross-view residual (quick, 1 subset/k)"
process_one "outputs/variable_views_crossview_residual_full.json" "full" "MPJPE@k cross-view residual (full, 5 subsets/k)"
process_one "outputs/variable_views_curriculum_final.json" "curriculum" "MPJPE@k curriculum (full, 5 subsets/k)"

echo "[postprocess] committing"
git add docs/figures/variable_views_*.png docs/results_variable_views_*.md || true
git commit -m "docs(results): variable-view MPJPE@k curves and tables" || true
git push origin multiview-residual-exploration || true

echo "[postprocess] done"
