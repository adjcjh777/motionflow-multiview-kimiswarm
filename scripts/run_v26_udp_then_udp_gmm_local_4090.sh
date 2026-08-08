#!/usr/bin/env bash
# Sequential local 4090 runs: v26+UDP then v26+UDP-GMM.
set -euo pipefail

echo "[$(date)] Starting v26+UDP local 4090 run..."
bash scripts/run_v26_udp_local_4090.sh

echo "[$(date)] Starting v26+UDP-GMM local 4090 run..."
bash scripts/run_v26_udp_gmm_local_4090.sh

echo "[$(date)] Both runs finished."
