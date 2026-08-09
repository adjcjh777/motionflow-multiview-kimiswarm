#!/usr/bin/env bash
# Local RTX 4090 nohup queue: v52 UWT medium -> v53 PSC medium -> v54 PSC-v2 medium.
#
# This script is meant to be launched with nohup so the training queue
# survives shell disconnects:
#   nohup bash scripts/nohup_v52_v53_v54_medium_queue_local_4090.sh > outputs/v52_v53_v54_medium_queue_nohup.log 2>&1 &
#
# It waits for the currently running v52 UWT medium smoke to finish,
# then runs v53 PSC medium, then v54 PSC-v2 medium.  GPU is assumed to be
# the single local RTX 4090, so runs are sequential.
set -euo pipefail

LOG="outputs/v52_v53_v54_medium_queue_local_4090_nohup.log"
exec > "$LOG" 2>&1

echo "[$(date -Iseconds)] Queue started.  Waiting for v52 UWT medium to finish..."

V52_LOG="outputs/omniview_fusion_v52_uwt_medium_local_4090.log"
while ! grep -q "Best val MPJPE" "$V52_LOG" 2>/dev/null; do
    sleep 60
done

echo "[$(date -Iseconds)] v52 UWT medium finished.  Running v53 PSC medium..."
bash scripts/run_v53_physical_space_calibration_medium_local_4090.sh

echo "[$(date -Iseconds)] v53 PSC medium finished.  Running v54 PSC-v2 medium..."
bash scripts/run_v54_psc_v2_medium_local_4090.sh

echo "[$(date -Iseconds)] Queue complete."
