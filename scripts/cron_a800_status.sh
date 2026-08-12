#!/usr/bin/env bash
# Cron-friendly wrapper for scripts/cron_a800_status.py.
#
# Usage (from crontab, running every 10 minutes):
#   */10 * * * * /path/to/this/repo/scripts/cron_a800_status.sh
#
# The script resolves its own location, then runs the Python monitor with
# stdout/stderr redirected to the same cron log file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG="${REPO_ROOT}/outputs/cron_a800_status.log"

mkdir -p "$(dirname "${LOG}")"

# Use Python 3 from WSL.  Adjust PYTHON if you want the project venv.
PYTHON="${PYTHON:-/usr/bin/python3}"

"${PYTHON}" -u "${SCRIPT_DIR}/cron_a800_status.py" >>"${LOG}" 2>&1
