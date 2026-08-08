#!/usr/bin/env bash
# Sequential local 4090 runs: v26+UDP, v26+UDP-GMM, v26+UDP+v28, v26+UDP-GMM+v28.
set -euo pipefail

echo "[$(date)] Starting v26+UDP local 4090 run..."
bash scripts/run_v26_udp_local_4090.sh

echo "[$(date)] Starting v26+UDP-GMM local 4090 run..."
bash scripts/run_v26_udp_gmm_local_4090.sh

echo "[$(date)] Starting v26+UDP+v28 local 4090 run..."
bash scripts/run_v26_udp_v28_local_4090.sh

echo "[$(date)] Starting v26+UDP-GMM+v28 local 4090 run..."
bash scripts/run_v26_udp_gmm_v28_local_4090.sh

echo "[$(date)] All runs finished."
