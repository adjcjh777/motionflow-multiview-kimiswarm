#!/usr/bin/env bash
# Train the cross-view residual model on the WebBridge Human3.6M corpus.
# Assumes `experiments/batch_convert_h36m_webbridge.py` has already produced
# canonical .npz files under data/webbridge/h36m.
#
# Example:
#   bash scripts/run_h36m_crossview_residual.sh

set -e

DATA_DIR="data/webbridge/h36m"

# Training set: subject 1, all actions (train split).
TRAIN_FILES=("$DATA_DIR"/s_01_acts_*_train_multiview.npz)

# Validation set: subject 5, action 2 (train split) as a held-out subject.
VAL_FILE="$DATA_DIR/s_05_acts_02_train_multiview.npz"

# Fallback if the train/val split naming is unavailable (legacy naming).
if [ ! -f "$VAL_FILE" ]; then
    VAL_FILE="$DATA_DIR/s_05_acts_02_multiview.npz"
fi

conda run -n mf python experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py \
    --train "${TRAIN_FILES[@]}" \
    --val "$VAL_FILE" \
    --clip_len 13 \
    --d 64 \
    --n_st_layers 2 \
    --residual_hidden 128 \
    --batch_size 8 \
    --train_samples 4000 \
    --epochs 30 \
    --reproj_weight 0.01 \
    --output outputs/crossview_residual_h36m.pth
