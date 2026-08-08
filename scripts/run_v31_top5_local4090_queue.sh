#!/usr/bin/env bash
# Sequential local RTX 4090 smoke queue for v31 top-5 ablations.
# The v30 smoke is run first as a val_stride=1 baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=(
    scripts/run_v30_smoke_local_4090_val1.sh
    scripts/launch_v31_domain_balanced_sampling_local4090.sh
    scripts/launch_v31_physical_floor_only_warmup_local4090.sh
    scripts/launch_v31_hierarchical_more_dropout_local4090.sh
    scripts/launch_v31_outlier_view_adaptive_threshold_local4090.sh
    scripts/launch_v31_epipolar_guided_sampling_local4090.sh
)

for run in "${RUNS[@]}"; do
    echo "$(date -Iseconds) Running ${run}"
    bash "${run}"
    echo "$(date -Iseconds) Finished ${run}"
done
