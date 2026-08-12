#!/usr/bin/env bash
# Launcher for the MVPose H36M true-GT v2 baseline.
#
# This script exports the v2 common format, converts to MVPose input,
# runs the geometry-only top-down triangulation adapter on the validation
# (test) split, and evaluates the predictions.
#
# The adapter runs on CPU (CUDA is disabled) so it does not compete for the
# project GPUs. It may still be run on A800 once a GPU is free by setting
# CUDA_VISIBLE_DEVICES appropriately.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/mvpose_h36m_true_gt_v2.yaml"

PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"

mkdir -p outputs/sota_baselines tmp/sota_baselines

# ---------------------------------------------------------------------------
# 1. Export H36M true-GT v2 to the common baseline format
# ---------------------------------------------------------------------------
INPUT_PKL="${REPO_ROOT}/tmp/sota_baselines/h36m_true_gt_v2_baseline_format.pkl"

if [[ ! -f "${INPUT_PKL}" ]]; then
    echo "[1/3] Exporting H36M true-GT v2 to common baseline format..."
    "${PYTHON}" scripts/sota_baselines/common_export_h36m_true_gt.py \
        --split_yaml configs/splits/h36m_true_gt_v2_standard.yaml \
        --output "${INPUT_PKL}"
else
    echo "[1/3] Common baseline format already exists: ${INPUT_PKL}"
fi

# ---------------------------------------------------------------------------
# 2. Convert common format to MVPose-specific input
# ---------------------------------------------------------------------------
echo "[2/3] Converting to MVPose input format..."
"${PYTHON}" scripts/sota_baselines/convert_to_mvpose_format.py \
    --config "${CONFIG}"

# ---------------------------------------------------------------------------
# 3. Run geometry-only triangulation adapter and evaluate
# ---------------------------------------------------------------------------
VAL_PKL="${REPO_ROOT}/tmp/sota_baselines/mvpose_data_v2/h36m_true_gt_val.pkl"
PRED_DIR="${REPO_ROOT}/tmp/sota_baselines/mvpose_predictions_v2"
METRICS_JSON="${REPO_ROOT}/outputs/sota_baselines/mvpose_h36m_true_gt_v2_metrics.json"

mkdir -p "${PRED_DIR}"

echo "[3/3] Running MVPose geometry-only adapter on ${VAL_PKL}..."
# Disable GPU to avoid touching the project GPUs.
CUDA_VISIBLE_DEVICES="" "${PYTHON}" -u scripts/sota_baselines/mvpose_h36m_adapter.py \
    --input_pkl "${VAL_PKL}" \
    --output_dir "${PRED_DIR}"

echo "Evaluating predictions..."
"${PYTHON}" scripts/sota_baselines/eval_mvpose_predictions.py \
    --input_pkl "${VAL_PKL}" \
    --pred_dir "${PRED_DIR}" \
    --out_json "${METRICS_JSON}" \
    --unit m

echo "[$(date -Iseconds)] MVPose H36M true-GT v2 run finished."
echo "Metrics: ${METRICS_JSON}"
