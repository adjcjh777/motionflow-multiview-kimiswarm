#!/usr/bin/env bash
# Self-evolution: wait for full pp model, evaluate, then run mixed-dataset training.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

CKPT=outputs/principal_point_pp_supervised_full.pth

if [[ ! -f "$CKPT" ]]; then
  echo "Waiting for $CKPT..."
  while [[ ! -f "$CKPT" ]]; do
    sleep 60
  done
  sleep 10
fi

echo "Evaluating $CKPT..."
python -u experiments/eval_principal_point_model_mpiinf3dhp.py \
  --checkpoint "$CKPT" \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 64 --residual_hidden 128 --principal_point_hidden 64 \
  --batch_size 8 --val_stride 10 \
  --out_json outputs/principal_point_pp_supervised_full_eval.json

echo "Starting mixed-dataset training..."
bash scripts/run_mixed_pp_wsl.sh
