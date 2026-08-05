#!/usr/bin/env bash
# Evaluate every Phase A checkpoint on the robustness benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

VAL=data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz

for ckpt in outputs/pp_ablation_A*.pth; do
  [[ -e "$ckpt" ]] || continue
  name=$(basename "$ckpt" .pth)
  echo "Evaluating $name..."
  python -u experiments/eval_principal_point_model_mpiinf3dhp.py \
    --checkpoint "$ckpt" \
    --dataset "$VAL" \
    --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
    --batch_size 8 --val_stride 10 \
    --out_json "outputs/${name}_eval.json"
done

echo "Phase A evaluation complete. Results:"
ls -1 outputs/pp_ablation_A*_eval.json || true
