#!/usr/bin/env bash
# Launch a local RTX 4090 training script under nohup and register it with the watchdog.
#
# Usage: bash scripts/local_4090_launch.sh <name> <launcher_script>
#   name               short tag, e.g. v46_svg_medium_local_4090
#   launcher_script    bash script to run, e.g. scripts/run_v46_svg_medium_local_4090.sh
#
# The wrapper writes a PID file to outputs/<name>.pid, which the watchdog uses to
# detect and restart the run if it dies.

set -euo pipefail

NAME="${1:?Usage: $0 <name> <launcher_script>}"
SCRIPT="${2:?Usage: $0 <name> <launcher_script>}"

NOHUP_LOG="outputs/${NAME}_nohup.log"
PID_FILE="outputs/${NAME}.pid"

# Redirect stdout/stderr to a nohup log.  The launcher script itself usually
# writes to its own log, so this wrapper log only captures shell messages.
nohup bash "${SCRIPT}" > "${NOHUP_LOG}" 2>&1 &
PID=$!

mkdir -p outputs
cat > "${PID_FILE}" <<EOF
PID=${PID}
SCRIPT=${SCRIPT}
NAME=${NAME}
EOF
# Mark this run as auto-restartable by the watchdog.
touch "outputs/${NAME}.autorestart"

echo "Launched ${NAME} (PID ${PID}) from ${SCRIPT}"
echo "PID file: ${PID_FILE}"
