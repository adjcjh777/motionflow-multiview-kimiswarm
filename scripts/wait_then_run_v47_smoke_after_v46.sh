#!/usr/bin/env bash
# Wait for the v46-SVG smoke to finish, then run the v47 temporal smoke.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while pgrep -f "v46_svg_smoke" >/dev/null; do
    echo "$(date -Iseconds) v46-SVG smoke still running; sleeping 60s"
    sleep 60
done

echo "$(date -Iseconds) v46-SVG smoke finished; waiting 120s for GPU memory release"
sleep 120

echo "$(date -Iseconds) launching v47 temporal smoke"
bash "${SCRIPT_DIR}/run_v47_temporal_svg_smoke_local_4090.sh"
