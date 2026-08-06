#!/usr/bin/env bash
# Run the remaining GPU experiments in sequence after the visibility-v2 training finishes.
set -e
cd "$(dirname "$0")/.."

echo "[queue] starting factorized ST+PP smoke"
bash scripts/run_factorized_pp_smoke_wsl.sh

echo "[queue] waiting for factorized smoke to finish..."
while pgrep -f "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py" > /dev/null; do
    sleep 60
done

echo "[queue] starting PP robust re-train (reproj + stronger PP supervision)"
bash scripts/run_crossview_pp_robust_retrain_wsl.sh

echo "[queue] waiting for PP robust re-train to finish..."
while pgrep -f "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py" > /dev/null; do
    sleep 60
done

echo "[queue] starting SSL pre-training on H36M"
bash scripts/run_ssl_pretrain_h36m_full_wsl.sh

echo "[queue] waiting for SSL pre-training to finish..."
while pgrep -f "pretrain_ray_attention_ssl.py" > /dev/null; do
    sleep 60
done

echo "[queue] starting spatiotemporal PP training"
bash scripts/run_spatiotemporal_principal_point_wsl.sh

echo "[queue] all remaining GPU experiments completed"
