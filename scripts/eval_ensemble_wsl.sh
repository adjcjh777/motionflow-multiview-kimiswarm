#!/usr/bin/env bash
# Ensemble evaluation of d=128 checkpoints on MPI-INF-3DHP validation.
# Only loads checkpoints that exist to avoid failing mid-run.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

MODEL="${1:-bayesian_tri_v2_pp}"
OUT_JSON="${2:-outputs/bayesian_tri_v2_ensemble_eval.json}"

# List of d=128 candidate checkpoints; only existing ones are passed.
CHECKPOINTS=(
    outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth
    outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth
    outputs/bayesian_tri_v2_full_data_mpiinf3dhp.pth
    outputs/epipolar_bias_v2_lite_pp_full_data_mpiinf3dhp.pth
)

ARGS=()
for ckpt in "${CHECKPOINTS[@]}"; do
    if [ -f "$ckpt" ]; then
        ARGS+=("--checkpoint" "$ckpt")
    fi
done

if [ ${#ARGS[@]} -eq 0 ]; then
    echo "No d=128 checkpoints found for ensemble." >&2
    exit 1
fi

python -u experiments/prototypes/eval_ensemble_checkpoints.py \
  --model "$MODEL" \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 128 --residual_hidden 256 --n_st_layers 3 \
  --batch_size 4 --val_stride 50 \
  --output_json "$OUT_JSON" \
  "${ARGS[@]}"
