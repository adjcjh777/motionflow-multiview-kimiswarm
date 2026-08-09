#!/usr/bin/env bash
# Run v46, v47, and v48 smoke runs sequentially on the local RTX 4090.
# Uses nohup so the chain survives shell disconnect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "$(date -Iseconds) starting v46-SVG smoke"
bash "${SCRIPT_DIR}/run_v46_svg_smoke_local_4090.sh"

echo "$(date -Iseconds) v46-SVG smoke done; sleeping 120s for GPU release"
sleep 120

echo "$(date -Iseconds) starting v47 temporal smoke"
bash "${SCRIPT_DIR}/run_v47_temporal_svg_smoke_local_4090.sh"

echo "$(date -Iseconds) v47 temporal smoke done; sleeping 120s for GPU release"
sleep 120

echo "$(date -Iseconds) starting v48 domain smoke"
bash "${SCRIPT_DIR}/run_v48_domain_smoke_local_4090.sh"

echo "$(date -Iseconds) all smoke runs complete"
