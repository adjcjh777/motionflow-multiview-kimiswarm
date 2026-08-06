#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for experiments/prototypes/run_extended_robustness_matrix.py.
# Defaults to CPU-only execution; pass "cuda" as the fourth argument to use GPU.
#
# Usage:
#     scripts/run_extended_robustness_matrix_wsl.sh <model> <checkpoint> <dataset> [device]
#
# Example:
#     scripts/run_extended_robustness_matrix_wsl.sh \
#         bayesian_tri_v2_pp \
#         outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.pth \
#         data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz

MODEL="${1:-bayesian_tri_v2_pp}"
CHECKPOINT="${2:-outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.pth}"
DATASET="${3:-data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz}"
DEVICE="${4:-cpu}"

export CUDA_VISIBLE_DEVICES=-1

OUT_DIR="outputs/extended_robustness_matrix_${MODEL}"
mkdir -p "$OUT_DIR"

python "experiments/prototypes/run_extended_robustness_matrix.py" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --output_dir "$OUT_DIR" \
    --clip_len 13 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --batch_size 2 \
    --val_stride 1 \
    --seed 42
