#!/usr/bin/env bash
# Run both OmniMultiViewFusionV2 robustness evaluations.
#
# Usage:
#   scripts/run_omniview_fusion_v2_robustness_wsl.sh
#   scripts/run_omniview_fusion_v2_robustness_wsl.sh --smoke
#   scripts/run_omniview_fusion_v2_robustness_wsl.sh /path/to/checkpoint /path/to/dataset
#
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=${MF_VENV:-.venv}
# shellcheck source=/dev/null
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

SMOKE_FLAG=""
CHECKPOINT="outputs/omniview_fusion_v2_d128_no_graph.pth"
DATASET="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"

# Parse positional/optional arguments.
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--smoke" ]]; then
        SMOKE_FLAG="--smoke"
        shift
    else
        CHECKPOINT="$1"
        shift
        if [[ $# -gt 0 ]]; then
            DATASET="$1"
            shift
        fi
    fi
fi

if [[ -n "$SMOKE_FLAG" ]] || [[ ! -f "$CHECKPOINT" ]] || [[ ! -f "$DATASET" ]]; then
    SMOKE_FLAG="--smoke"
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Running in smoke mode: checkpoint $CHECKPOINT not found."
    elif [[ ! -f "$DATASET" ]]; then
        echo "Running in smoke mode: dataset $DATASET not found."
    else
        echo "Running in smoke mode (--smoke passed)."
    fi
fi

PYTHON="python -u"

echo "============================================"
echo "OmniMultiViewFusionV2 Variable-View Robustness"
echo "============================================"
$PYTHON experiments/eval_omniview_fusion_v2_variable_views.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    $SMOKE_FLAG \
    --out_json outputs/eval_omniview_fusion_v2_variable_views.json

echo ""
echo "============================================"
echo "OmniMultiViewFusionV2 Camera Perturbation Robustness"
echo "============================================"
$PYTHON experiments/eval_omniview_fusion_v2_camera_perturbation.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    $SMOKE_FLAG \
    --out_json outputs/eval_omniview_fusion_v2_camera_perturbation.json

echo ""
echo "All robustness evaluations complete."
