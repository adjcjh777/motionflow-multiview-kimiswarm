#!/usr/bin/env bash
# Sequential clean smoke pair: v25 then v81 true-GT H36M on local RTX 4090.
# Each smoke uses the identical tiny 2-epoch/256-sample schedule defined by the
# individual smoke scripts. Runs serially because only one local training task
# is allowed at a time.
set -euo pipefail

WRAPPER_LOG="outputs/v25_v81_smoke_pair_wrapper.log"
: > "$WRAPPER_LOG"

echo "[$(date -Iseconds)] Starting v25 smoke" >> "$WRAPPER_LOG"
bash scripts/run_v25_true_gt_h36m_smoke_local_4090.sh
echo "[$(date -Iseconds)] v25 smoke finished" >> "$WRAPPER_LOG"

echo "[$(date -Iseconds)] Starting v81 smoke" >> "$WRAPPER_LOG"
bash scripts/run_v81_true_gt_h36m_smoke_local_4090.sh
echo "[$(date -Iseconds)] v81 smoke finished" >> "$WRAPPER_LOG"

echo "[$(date -Iseconds)] Both smokes complete" >> "$WRAPPER_LOG"
