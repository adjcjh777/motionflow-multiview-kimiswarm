#!/usr/bin/env bash
# Run after the v46/v47/v48 smoke chain finishes on the local RTX 4090.
# Usage: bash scripts/run_after_smoke_chain_local_4090.sh <pid-of-smoke-chain>
set -euo pipefail

CHAIN_PID="${1:-}"
if [[ -z "$CHAIN_PID" ]]; then
    echo "Usage: $0 <pid-of-smoke-chain>"
    exit 1
fi

echo "$(date -Iseconds) waiting for smoke chain PID $CHAIN_PID to finish"
while kill -0 "$CHAIN_PID" 2>/dev/null; do
    sleep 60
done
echo "$(date -Iseconds) smoke chain finished; starting v49 ablation matrix"
bash scripts/run_v49_ablation_matrix_local_4090.sh
echo "$(date -Iseconds) v49 ablation matrix finished; starting v50 SEFH smoke"
bash scripts/run_v50_self_evolution_feedback_head_smoke_local_4090.sh
echo "$(date -Iseconds) v50 SEFH smoke finished; starting v51 DAE smoke"
bash scripts/run_v51_domain_agnostic_ensemble_smoke_local_4090.sh
echo "$(date -Iseconds) all post-chain runs complete"
