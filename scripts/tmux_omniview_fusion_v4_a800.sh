#!/usr/bin/env bash
# Launch ``scripts/run_omniview_fusion_v4_a800.sh`` inside a persistent tmux
# session on A800-D, with a lock file to prevent duplicate sessions.
#
# The wrapper itself is intentionally thin: it checks pre-conditions, kills any
# stale session of the same name, starts a new detached tmux session, and exits.
# The actual training resilience (GPU selection, nohup, restarts) lives in
# ``run_omniview_fusion_v4_a800.sh``.
#
# Usage:
#     # Normal launch
#     bash scripts/tmux_omniview_fusion_v4_a800.sh
#
#     # Dry-run (print what would be launched, do not touch tmux)
#     bash scripts/tmux_omniview_fusion_v4_a800.sh --dry-run
#
#     # Override session/experiment name
#     MF_EXPERIMENT_NAME=v4_ablation bash scripts/tmux_omniview_fusion_v4_a800.sh
#
set -euo pipefail

ROOT="${MF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXPERIMENT_NAME="${MF_EXPERIMENT_NAME:-omniview_fusion_v4}"
SESSION_NAME="${MF_SESSION_NAME:-${EXPERIMENT_NAME}}"
RUN_SCRIPT="${ROOT}/scripts/run_omniview_fusion_v4_a800.sh"
LOCK_DIR="${MF_LOCK_DIR:-${ROOT}/tmp}"
LOCK_FILE="${LOCK_DIR}/${EXPERIMENT_NAME}.tmux.lock"
DRY_RUN=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --help)
            sed -n '2,/^# Usage/s/^# \{0,1\}//p' "$0"
            exit 0
            ;;
        *)
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ ! -x "$RUN_SCRIPT" ]]; then
    chmod +x "$RUN_SCRIPT" 2>/dev/null || true
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "ERROR: Runner script not found: $RUN_SCRIPT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Lock file: prevent duplicate tmux sessions for the same experiment.
# ---------------------------------------------------------------------------
mkdir -p "$LOCK_DIR"
if [[ -f "$LOCK_FILE" ]]; then
    pid=$(cat "$LOCK_FILE" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: tmux session '$SESSION_NAME' already running (PID $pid)." >&2
        echo "Attach: tmux attach -t $SESSION_NAME" >&2
        exit 1
    fi
    rm -f "$LOCK_FILE"
fi

# ---------------------------------------------------------------------------
# Dry-run: only print the command that would be executed.
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] Would start tmux session '$SESSION_NAME'"
    echo "[dry-run] Lock file: $LOCK_FILE"
    echo "[dry-run] Runner:    $RUN_SCRIPT"
    echo "[dry-run] Command:   tmux new-session -d -s $SESSION_NAME bash $RUN_SCRIPT"
    exit 0
fi

# ---------------------------------------------------------------------------
# Ensure tmux is available.
# ---------------------------------------------------------------------------
TMUX_BIN=""
for candidate in tmux /usr/bin/tmux /usr/local/bin/tmux; do
    if command -v "$candidate" >/dev/null 2>&1; then
        TMUX_BIN="$candidate"
        break
    fi
done

if [[ -z "$TMUX_BIN" ]]; then
    echo "ERROR: tmux not found on PATH." >&2
    exit 1
fi

# Kill any stale session with the same name.
if "$TMUX_BIN" has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Removing stale tmux session '$SESSION_NAME'"
    "$TMUX_BIN" kill-session -t "$SESSION_NAME" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Start the tmux session.
# The runner itself uses nohup internally; this wrapper just keeps the session
# alive and records the lock.
# ---------------------------------------------------------------------------
echo "Starting tmux session '$SESSION_NAME' for $EXPERIMENT_NAME"
echo "Lock file: $LOCK_FILE"

"$TMUX_BIN" new-session -d -s "$SESSION_NAME" "bash -lc 'echo \$\$ > $LOCK_FILE; bash $RUN_SCRIPT; rm -f $LOCK_FILE'"

# Verify the session started.
sleep 0.5
if "$TMUX_BIN" has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' started."
    echo "Attach:  tmux attach -t $SESSION_NAME"
    echo "Logs:    ${ROOT}/outputs/${EXPERIMENT_NAME}.log"
else
    echo "ERROR: tmux session '$SESSION_NAME' failed to start." >&2
    exit 1
fi
