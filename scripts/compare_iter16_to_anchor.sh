#!/usr/bin/env bash
# Compare iter16 full-run checkpoints against the 8.75 mm anchor.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

ANCHOR_CKPT=outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth
DATASET=data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz

echo "=== Anchor: $ANCHOR_CKPT ==="
python -u experiments/eval_full_metrics.py \
    --model crossview_residual_pp \
    --dataset "$DATASET" \
    --checkpoint "$ANCHOR_CKPT" \
    --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 --val_stride 50

echo ""
echo "=== Gaussian-Splatting: $GAUSS_CKPT ==="
GAUSS_CKPT=outputs/splat_pp_full_mpiinf3dhp.pth
if [ -f "$GAUSS_CKPT" ]; then
    python -u experiments/eval_full_metrics.py \
        --model splat_pp \
        --dataset "$DATASET" \
        --checkpoint "$GAUSS_CKPT" \
        --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 --val_stride 50
else
    echo "Checkpoint not found: $GAUSS_CKPT"
fi

echo ""
echo "=== Kinematic-Chain: $KC_CKPT ==="
KC_CKPT=outputs/kinematic_chain_pp_full_mpiinf3dhp.pth
if [ -f "$KC_CKPT" ]; then
    python -u experiments/eval_full_metrics.py \
        --model kinematic_chain_pp \
        --dataset "$DATASET" \
        --checkpoint "$KC_CKPT" \
        --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 --val_stride 50
else
    echo "Checkpoint not found: $KC_CKPT"
fi
