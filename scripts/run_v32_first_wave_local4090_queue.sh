#!/usr/bin/env bash
# First-wave v32 local RTX 4090 smoke queue.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=(
    scripts/run_v32_domain_aware_view_curriculum_smoke_local4090.sh
    scripts/run_v32_trajectory_consistency_smoke_local4090.sh
    scripts/run_v32_ray_attention_smoke_local4090.sh
    scripts/run_v32_physical_alignment_smoke_local4090.sh
)

for run in "${RUNS[@]}"; do
    echo "$(date -Iseconds) Running ${run}"
    bash "${run}"
    echo "$(date -Iseconds) Finished ${run}"
done
