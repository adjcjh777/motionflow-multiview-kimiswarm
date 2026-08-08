#!/usr/bin/env bash
# Launch the v25 small local 4090 baseline with a real-time CSV/JSON monitor.
#
# Usage:
#   bash scripts/monitor_v25_local_4090.sh          # full 20-epoch / 500-sample run
#   EPOCHS=2 TRAIN_SAMPLES=50 bash scripts/monitor_v25_local_4090.sh  # quick smoke
#
# To keep it alive after the shell exits:
#   bash scripts/nohup_monitor_v25_local_4090.sh
#
set -euo pipefail

TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT=${OUTPUT:-outputs/omniview_fusion_v25_geometry_fusion_small_local_4090_${TIMESTAMP}.pth}
LOG=${LOG:-outputs/omniview_fusion_v25_geometry_fusion_small_local_4090_${TIMESTAMP}.log}
CSV=${CSV:-outputs/v25_local_4090_monitor.csv}
STATUS=${STATUS:-outputs/v25_local_4090_status.json}
EPOCHS=${EPOCHS:-20}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-500}
BATCH_SIZE=${BATCH_SIZE:-16}

mkdir -p outputs

echo "[$(date)] Launching v25 local 4090 baseline"
echo "  epochs=$EPOCHS train_samples=$TRAIN_SAMPLES batch_size=$BATCH_SIZE"
echo "  output=$OUTPUT"
echo "  log=$LOG"
echo "  csv=$CSV"
echo "  status=$STATUS"

# Start the training in the background so the monitor can tail the log immediately.
EPOCHS=$EPOCHS TRAIN_SAMPLES=$TRAIN_SAMPLES BATCH_SIZE=$BATCH_SIZE OUTPUT=$OUTPUT LOG=$LOG bash scripts/run_v25_small_local_4090.sh &
TRAIN_PID=$!
echo "$TRAIN_PID" > outputs/v25_local_4090_train.pid
echo "[$(date)] Training started PID=$TRAIN_PID log=$LOG"

# Start the monitor in the background; it will exit when the training PID is gone.
python scripts/v25_local_monitor.py \
    --log "$LOG" \
    --csv "$CSV" \
    --status "$STATUS" \
    --pid $TRAIN_PID \
    --poll-interval 10 &
MON_PID=$!
echo "$MON_PID" > outputs/v25_local_4090_monitor.pid
echo "[$(date)] Monitor started PID=$MON_PID csv=$CSV"
