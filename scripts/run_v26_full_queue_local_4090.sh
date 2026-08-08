#!/usr/bin/env bash
# Sequential full-scale local 4090 runs: v26+UDP, v26+UDP-GMM, v26+UDP+v28, v26+UDP-GMM+v28.
set -euo pipefail

echo "[$(date)] Starting v26+UDP full local 4090 run..."
bash scripts/run_v26_udp_full_local_4090.sh

echo "[$(date)] Starting v26+UDP-GMM full local 4090 run..."
bash scripts/run_v26_udp_gmm_full_local_4090.sh

echo "[$(date)] Starting v26+UDP+v28 full local 4090 run..."
bash scripts/run_v26_udp_v28_full_local_4090.sh

echo "[$(date)] Starting v26+UDP-GMM+v28 full local 4090 run..."
bash scripts/run_v26_udp_gmm_v28_full_local_4090.sh

echo "[$(date)] All full runs finished."
