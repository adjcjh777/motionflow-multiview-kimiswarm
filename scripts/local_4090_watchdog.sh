#!/usr/bin/env bash
# Lightweight watchdog for local RTX 4090 nohup training runs.
#
# WSL does not have tmux, so we use nohup.  This script scans outputs/*.pid,
# checks whether the recorded PID is still alive, and relaunches dead runs
# that have a matching *.autorestart flag and no *.stop flag.
#
# Intended to be run every few minutes via cron or a nohup loop, e.g.
#   while true; do bash scripts/local_4090_watchdog.sh; sleep 300; done

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs
WATCHDOG_LOG="outputs/local_4090_watchdog.log"

touch "${WATCHDOG_LOG}"

for pid_file in outputs/*.pid; do
    [ -e "${pid_file}" ] || continue
    name=$(basename "${pid_file}" .pid)
    autorestart="outputs/${name}.autorestart"
    stop_flag="outputs/${name}.stop"

    if [ ! -f "${autorestart}" ]; then
        continue
    fi
    if [ -f "${stop_flag}" ]; then
        continue
    fi

    # shellcheck source=/dev/null
    source "${pid_file}"

    if kill -0 "${PID}" 2>/dev/null; then
        continue
    fi

    echo "$(date -Iseconds) ${name}: PID ${PID} dead; relaunching from ${SCRIPT}" >> "${WATCHDOG_LOG}"
    nohup bash "${SCRIPT}" > "outputs/${name}_nohup.log" 2>&1 &
    new_pid=$!
    cat > "${pid_file}" <<EOF
PID=${new_pid}
SCRIPT=${SCRIPT}
NAME=${name}
EOF
    echo "$(date -Iseconds) ${name}: relaunched as PID ${new_pid}" >> "${WATCHDOG_LOG}"
done
