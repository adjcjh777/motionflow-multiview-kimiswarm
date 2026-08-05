#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

echo "[postprocess] waiting for variable-view eval to finish..."
while pgrep -f "experiments/eval_variable_views.py" > /dev/null; do
    sleep 60
done

echo "[postprocess] plotting variable-view curve"
python experiments/plot_variable_views.py \
    --input outputs/variable_views_crossview_residual_full.json \
    --output docs/figures/variable_views_crossview_residual_full.png \
    --title "MPJPE@k cross-view residual MPI S2 full"

echo "[postprocess] updating docs/results_variable_views.md"
python - <<'PY'
import json
import datetime
from pathlib import Path

p = Path('outputs/variable_views_crossview_residual_full.json')
out = Path('docs/results_variable_views.md')
if p.exists():
    data = json.load(open(p))
    lines = ['# Variable-View MPJPE@k Results\n\n',
             f'Updated: {datetime.datetime.utcnow().isoformat()}Z\n\n']
    lines.append('| k | mean_mm | std_mm | subsets |\n')
    lines.append('|---|---|---|---|\n')
    for k in sorted(data.keys(), key=int):
        v = data[k]
        lines.append(f"| {k} | {v['mean_mm']:.4f} | {v['std_mm']:.4f} | {v['n_subsets']} |\n")
    out.write_text(''.join(lines))
    print(f'Wrote {out}')
PY"

echo "[postprocess] committing"
git add docs/figures/variable_views_crossview_residual_full.png docs/results_variable_views.md || true
git commit -m "docs(results): variable-view MPJPE@k curve and table" || true
git push origin multiview-residual-exploration || true

echo "[postprocess] done"
