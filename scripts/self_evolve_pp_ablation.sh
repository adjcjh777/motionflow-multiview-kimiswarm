#!/usr/bin/env bash
# Self-evolving ablation loop:
# 1. Wait for the current small pp-supervised run to finish.
# 2. Evaluate it.
# 3. Run Phase A ablations sequentially.
# 4. Summarize results.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

CKPT=outputs/principal_point_pp_supervised_small_v2.pth

# Step 1: Wait for the current training run to finish (poll the checkpoint file).
if [[ ! -f "$CKPT" ]]; then
  echo "Waiting for $CKPT to appear..."
  while [[ ! -f "$CKPT" ]]; do
    sleep 60
  done
  # Give the writer a moment to finish flushing.
  sleep 10
fi

# Step 2: Evaluate the checkpoint.
echo "Evaluating $CKPT..."
python -u experiments/eval_principal_point_model_mpiinf3dhp.py \
  --checkpoint "$CKPT" \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
  --batch_size 8 --val_stride 10 \
  --out_json outputs/principal_point_pp_supervised_small_v2_eval.json

# Step 3: Run Phase A ablations sequentially.
echo "Starting Phase A ablations..."
bash scripts/phase_a_ablation_runner.sh

# Step 4: Evaluate all Phase A checkpoints.
bash scripts/phase_a_eval_runner.sh

# Step 5: Summarize.
echo "Ablation complete. Summarizing results..."
python - <<'PY'
import json
import glob
from pathlib import Path

for path in sorted(glob.glob("outputs/pp_ablation_A*_eval.json")):
    data = json.loads(Path(path).read_text())
    name = Path(path).stem
    clean = data.get("clean", {}).get("mpjpe", float("nan"))
    cxcy3 = data.get("cxcy_3px", {}).get("mpjpe", float("nan"))
    cxcy5 = data.get("cxcy_5px", {}).get("mpjpe", float("nan"))
    print(f"{name}: clean={clean:.2f}mm  cxcy_3px={cxcy3:.2f}mm  cxcy_5px={cxcy5:.2f}mm")

# Also show current small run if available.
p = Path("outputs/principal_point_pp_supervised_small_v2_eval.json")
if p.exists():
    data = json.loads(p.read_text())
    clean = data.get("clean", {}).get("mpjpe", float("nan"))
    cxcy3 = data.get("cxcy_3px", {}).get("mpjpe", float("nan"))
    cxcy5 = data.get("cxcy_5px", {}).get("mpjpe", float("nan"))
    print(f"small_v2: clean={clean:.2f}mm  cxcy_3px={cxcy3:.2f}mm  cxcy_5px={cxcy5:.2f}mm")
PY
