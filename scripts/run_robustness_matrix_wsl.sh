#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for experiments/run_robustness_matrix.py.
# Defaults to CPU-only execution; pass "cuda" as the fourth argument to use GPU.
#
# Usage:
#     scripts/run_robustness_matrix_wsl.sh <model> <checkpoint> <dataset> [device]
#
# Example:
#     scripts/run_robustness_matrix_wsl.sh \
#         epipolar_bias_v2_pp \
#         outputs/epipolar_bias_v2_smoke.pth \
#         tmp/mpi_s02_seq01_smoke.npz

MODEL="${1:-epipolar_bias_v2_pp}"
CHECKPOINT="${2:-outputs/epipolar_bias_v2_smoke.pth}"
DATASET="${3:-tmp/mpi_s02_seq01_smoke.npz}"
DEVICE="${4:-cpu}"

export CUDA_VISIBLE_DEVICES=-1

OUT_DIR="outputs/robustness_matrix_${MODEL}"
mkdir -p "$OUT_DIR"

python "experiments/run_robustness_matrix.py" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --output_dir "$OUT_DIR" \
    --clip_len 5 \
    --d 32 \
    --residual_hidden 64 \
    --n_st_layers 2 \
    --batch_size 2 \
    --val_stride 1 \
    --seed 42
