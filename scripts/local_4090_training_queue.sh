#!/usr/bin/env bash
# Sequential 4090 training queue. Each job is started only after the previous one finishes.
set -euo pipefail

# Job 1: v25 small baseline
export EPOCHS=${EPOCHS:-20}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-500}
export BATCH_SIZE=${BATCH_SIZE:-16}

echo "[QUEUE] Starting v25 small local baseline"
bash scripts/run_v25_small_local_4090.sh

# Job 2: v25 + v18 top-k straight-through
export OUTPUT=outputs/omniview_fusion_v25_v18_topk_st_small_local_4090.pth
export LOG=outputs/omniview_fusion_v25_v18_topk_st_small_local_4090.log
echo "[QUEUE] Starting v25 + v18 top-k ST"
bash scripts/run_v25_v18_topk_st_local_4090.sh

# Job 3: add more variants here
# bash scripts/run_v25_xxx_local_4090.sh

echo "[QUEUE] All local 4090 jobs finished"
