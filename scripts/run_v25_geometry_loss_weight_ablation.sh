#!/usr/bin/env bash
# v25 geometry-loss weight ablation runner (0.01, 0.1, 1.0)
# Runs the three small-schedule variants defined in
# configs/ablations/v25_geometry_loss_weight_ablation.yaml.
#
# Usage:
#   bash scripts/run_v25_geometry_loss_weight_ablation.sh          # run for real
#   DRY_RUN=1 bash scripts/run_v25_geometry_loss_weight_ablation.sh  # dry-run

set -euo pipefail

CONFIG="configs/ablations/v25_geometry_loss_weight_ablation.yaml"
DRY_RUN_FLAG=""
if [ "${DRY_RUN:-0}" = "1" ]; then
    DRY_RUN_FLAG="--dry-run"
fi

PYTHON=${PYTHON:-python}

$PYTHON -u scripts/run_ablation_variants.py \
    --config "$CONFIG" \
    $DRY_RUN_FLAG \
    --max-workers 1
