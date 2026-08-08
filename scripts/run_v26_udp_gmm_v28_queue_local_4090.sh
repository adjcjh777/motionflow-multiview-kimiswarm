#!/usr/bin/env bash
# Sequential local 4090 runs: v26+UDP-GMM, v26+UDP+v28, v26+UDP-GMM+v28.
# (Skips v26+UDP, which finished/overfit in earlier queue.)
set -euo pipefail

echo "[$(date)] Starting v26+UDP-GMM full local 4090 run..."
bash scripts/run_v26_udp_gmm_full_local_4090.sh

echo "[$(date)] Starting v26+UDP+v28 full local 4090 run..."
bash scripts/run_v26_udp_v28_full_local_4090.sh

echo "[$(date)] Starting v26+UDP-GMM+v28 full local 4090 run..."
bash scripts/run_v26_udp_gmm_v28_full_local_4090.sh

echo "[$(date)] All runs finished. Running post-queue benchmark..."
bash scripts/benchmark_v26_full_queue_local_4090.sh

echo "[$(date)] Post-queue benchmark finished."
