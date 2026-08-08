#!/usr/bin/env bash
# Smoke test for adaptive view selector tuning.
# Runs 1 epoch on synthetic data with the v5 model + adaptive view selection.
# Expected runtime: <2 min on RTX 4090, <5 min on CPU.

set -euo pipefail

cd "$(dirname "$0")/.."

python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_adaptive_view_selection true \
    --adaptive_view_k 2 \
    --adaptive_view_budget_weight 0.01 \
    --budget_loss_weight 0.1 \
    --output outputs/adaptive_view_selector_tuning_smoke.pth \
    --lr 1e-3 "$@"
