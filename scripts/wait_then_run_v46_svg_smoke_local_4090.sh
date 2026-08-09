#!/usr/bin/env bash
# Wait for the current v45-AGF medium local run to finish, then run the v46-SVG smoke.
# This is a nohup-friendly launcher so the smoke starts automatically once GPU is free.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V45_PID=1444

while kill -0 "$V45_PID" 2>/dev/null; do
    echo "$(date -Iseconds) v45-AGF medium still running (PID $V45_PID); sleeping 60s"
    sleep 60
done

echo "$(date -Iseconds) v45-AGF medium finished; waiting 120s for GPU memory release"
sleep 120

echo "$(date -Iseconds) launching v46-SVG smoke"
bash "${SCRIPT_DIR}/run_v46_svg_smoke_local_4090.sh"
