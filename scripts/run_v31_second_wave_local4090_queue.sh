#!/usr/bin/env bash
# Second-wave local RTX 4090 smoke queue for v31 variants that need source wiring.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=(
    scripts/run_v31_geometry_attention_refinement_smoke_local4090.sh
    scripts/run_v31_camera_view_embedding_smoke_local4090.sh
    scripts/run_v31_physical_collision_penalty_smoke_local4090.sh
)

for run in "${RUNS[@]}"; do
    echo "$(date -Iseconds) Running ${run}"
    bash "${run}"
    echo "$(date -Iseconds) Finished ${run}"
done
