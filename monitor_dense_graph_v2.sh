#!/bin/bash
set -o pipefail

REPO_PATH="D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm"
LOG_FILE="$REPO_PATH/outputs/omniview_fusion_v2_d128_dense_graph_v2.log"
GH_BIN="/tmp/gh/bin/gh.exe"
REPO="adjcjh777/motionflow-multiview-kimiswarm"
ISSUE="74"
SUMMARY_FILE="$REPO_PATH/monitor_dense_graph_v2_summary.txt"
STATUS_FILE="$REPO_PATH/monitor_dense_graph_v2_status.txt"

# Duration in seconds: 3 hours 45 minutes (3-4 hours)
END_TIME=$(($(date +%s) + 13500))
INTERVAL=720  # 12 minutes average, within 10-15 minute range

# Track state
LAST_EPOCH=""
LAST_LINE_COUNT=0
CHECK_COUNT=0
START_TIME=$(date +%s)

# Get GitHub token once
GH_TOKEN=""
if command -v git >/dev/null 2>&1; then
    GH_TOKEN=$(cd "$REPO_PATH" && printf 'url=https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git\n\n' | git credential fill 2>/dev/null | awk -F= '/^password=/{print $2}')
fi
export GH_TOKEN

post_issue() {
    local body="$1"
    if [ -n "$GH_TOKEN" ] && [ -f "$GH_BIN" ]; then
        "$GH_BIN" issue comment "$ISSUE" --repo "$REPO" --body "$body"
    else
        echo "Would post to issue #$ISSUE: $body" >> "$STATUS_FILE"
    fi
}

log_status() {
    local msg="$1"
    echo "[$(date -Iseconds)] $msg" >> "$STATUS_FILE"
    echo "[$(date -Iseconds)] $msg"
}

# Initialize status file
echo "Monitoring started at $(date -Iseconds)" > "$STATUS_FILE"

POSTED_STOP=false

while [ $(date +%s) -lt $END_TIME ]; do
    CHECK_COUNT=$((CHECK_COUNT + 1))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    log_status "Check #$CHECK_COUNT after ${ELAPSED}s"

    # Read latest epoch line from log
    if [ -f "$LOG_FILE" ]; then
        CURRENT_LINE_COUNT=$(wc -l < "$LOG_FILE")
        LATEST_EPOCH=$(grep -E '\[Freeze\] Epoch [0-9]+:|Epoch [0-9]+:' "$LOG_FILE" | tail -1)
        log_status "Log lines: $CURRENT_LINE_COUNT | Latest epoch: $LATEST_EPOCH"
        LAST_LINE_COUNT=$CURRENT_LINE_COUNT
        if [ -n "$LATEST_EPOCH" ]; then
            LAST_EPOCH="$LATEST_EPOCH"
        fi
    else
        log_status "Log file not found: $LOG_FILE"
        LATEST_EPOCH=""
    fi

    # Check nvidia-smi
    NVIDIA_INFO=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
    log_status "GPU: $NVIDIA_INFO"

    # Check for training process
    TRAIN_PID=$(ps -ef | grep -i "train_omniview_fusion_v2" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$TRAIN_PID" ]; then
        log_status "Training process alive PID: $TRAIN_PID"
        GPU_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E "^$TRAIN_PID$" | head -1)
        if [ -n "$GPU_PID" ]; then
            log_status "Training process using GPU: PID $GPU_PID"
        else
            log_status "WARNING: Training process found but not listed in nvidia-smi compute apps"
            if [ "$POSTED_STOP" = false ]; then
                post_issue "Training process PID $TRAIN_PID is alive but no longer using GPU at $(date -Iseconds). Latest epoch: ${LAST_EPOCH:-N/A}"
                POSTED_STOP=true
            fi
        fi
    else
        log_status "WARNING: Training process not found"
        if [ "$POSTED_STOP" = false ]; then
            post_issue "Training process not found at $(date -Iseconds). Latest epoch: ${LAST_EPOCH:-N/A}. Monitoring stopped/crashed/finished."
            POSTED_STOP=true
        fi
    fi

    # Detect finish or crash based on log
    if [ -f "$LOG_FILE" ]; then
        if tail -20 "$LOG_FILE" | grep -qiE "(finished|completed|done|saved checkpoint|training complete)"; then
            log_status "Training appears to have finished"
            if [ "$POSTED_STOP" = false ]; then
                post_issue "Training appears to have finished at $(date -Iseconds). Latest epoch: ${LAST_EPOCH:-N/A}"
                POSTED_STOP=true
            fi
        fi
        if tail -50 "$LOG_FILE" | grep -qiE "(error|exception|traceback|killed|cuda out of memory)"; then
            log_status "Possible crash/error detected in log"
            if [ "$POSTED_STOP" = false ]; then
                post_issue "Possible training crash/error detected at $(date -Iseconds). Latest epoch: ${LAST_EPOCH:-N/A}. Check logs."
                POSTED_STOP=true
            fi
        fi
    fi

    # Sleep for interval
    sleep $INTERVAL
done

# Final summary
echo "" >> "$STATUS_FILE"
echo "===== SUMMARY =====" >> "$STATUS_FILE"
echo "Total checks: $CHECK_COUNT" >> "$STATUS_FILE"
echo "Last epoch: ${LAST_EPOCH:-N/A}" >> "$STATUS_FILE"
echo "Monitoring ended at $(date -Iseconds)" >> "$STATUS_FILE"

cat "$STATUS_FILE"
