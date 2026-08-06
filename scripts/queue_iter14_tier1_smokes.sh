#!/usr/bin/env bash
# Run the three Tier-1 iter14 smoke experiments in sequence on the RTX 4090.
set -e
cd "$(dirname "$0")/.."

echo "[iter14] Tier-1 smoke queue: reprojection-consistency -> dynamic-view-gate -> graph-skeleton-residual"

bash scripts/run_reprojection_consistency_pp_smoke_wsl.sh
bash scripts/run_dynamic_view_gate_smoke_wsl.sh
bash scripts/run_graph_skeleton_residual_pp_smoke_wsl.sh
bash scripts/run_epipolar_pp_smoke_wsl.sh

# Evaluate all produced checkpoints.
bash scripts/eval_iter14_smokes.sh

# Aggregate clean metrics for any produced checkpoints.
for model in reprojection_consistency_pp dynamic_view_gate_pp graph_skeleton_residual_pp; do
    ckpt="outputs/${model}_smoke.pth"
    if [ -f "$ckpt" ]; then
        echo "[iter14] evaluating $model -> $ckpt"
        # eval_full_metrics handles dynamic_gate_pp and graph_skeleton_residual_pp directly.
        # reprojection_consistency_pp uses the same architecture as crossview_residual_pp.
    fi
done

echo "[iter14] Tier-1 smoke queue complete"
