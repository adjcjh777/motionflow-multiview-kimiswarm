#!/usr/bin/env bash
# Monitor all omniview training tmux sessions on A800.
# Lists sessions, tails the last 5 lines of each training log, and reports GPU usage.

set -uo pipefail

PROJECT_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "$PROJECT_ROOT" || { echo "Failed to cd to $PROJECT_ROOT"; exit 1; }

# Map tmux session names to their log files.
declare -A LOG_MAP=(
    ["v10_aleatoric_outlier_a800"]="outputs/omniview_fusion_v10_aleatoric_outlier.log"
    ["v10_no_outlier"]="outputs/omniview_fusion_v10_no_outlier.log"
    ["v11_irls"]="outputs/omniview_fusion_v11_irls.log"
    ["v12_adaptive_multiscale"]="outputs/omniview_fusion_v12_adaptive_multiscale.log"
)

echo "========================================"
echo "A800 Omniview Training Monitor"
echo "Timestamp: $(date -Iseconds)"
echo "========================================"
echo

# 1. List tmux sessions.
echo "--- Tmux Sessions ---"
tmux ls 2>/dev/null || echo "No tmux sessions found."
echo

# 2. For each known session, report status and last log lines.
echo "--- Session / Log Status ---"
for session in "${!LOG_MAP[@]}"; do
    log="${LOG_MAP[$session]}"
    echo "----------------------------------------"
    echo "Session: $session"
    echo "Log: $log"

    # Check if the tmux session is alive.
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Status: RUNNING"
    else
        echo "Status: CRASHED / NOT RUNNING"
    fi

    # Show the last 5 lines of the log if it exists.
    if [[ -f "$log" ]]; then
        echo "Last 5 log lines:"
        tail -n 5 "$log" | sed 's/^/  /'
    else
        echo "Log file not found."
    fi
    echo
done

# 3. GPU usage summary.
echo "--- GPU Usage ---"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv
echo

# 4. Process-level GPU usage for the training script.
echo "--- Processes on GPUs ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_name --format=csv | sed '1!s/^/  /'

echo "========================================"
echo "Monitor run complete: $(date -Iseconds)"
echo "========================================"
