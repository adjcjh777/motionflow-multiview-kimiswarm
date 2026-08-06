#!/usr/bin/env bash
set -euo pipefail

# CPU smoke test for RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2Lite.
export CUDA_VISIBLE_DEVICES=-1

LOG="outputs/epipolar_bias_v2_lite_pp_smoke.log"
mkdir -p "$(dirname "$LOG")"

python tests/test_epipolar_bias_v2_lite.py \
    2>&1 | tee "$LOG"
