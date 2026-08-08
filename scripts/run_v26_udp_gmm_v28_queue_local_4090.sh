#!/usr/bin/env bash
# Sequential local 4090 runs: v26+UDP-GMM, v26+UDP+v28, v26+UDP-GMM+v28.
# (Skips v26+UDP, which finished/overfit in earlier queue.)
# Each variant is allowed to fail without stopping the queue.
set -uo pipefail

echo "[$(date)] Starting v26+UDP-GMM full local 4090 run..."
bash scripts/run_v26_udp_gmm_full_local_4090.sh || echo "[WARN] v26+UDP-GMM failed or stopped early"

echo "[$(date)] Starting v26+UDP+v28 full local 4090 run..."
bash scripts/run_v26_udp_v28_full_local_4090.sh || echo "[WARN] v26+UDP+v28 failed or stopped early"

echo "[$(date)] Starting v26+UDP-GMM+v28 full local 4090 run..."
bash scripts/run_v26_udp_gmm_v28_full_local_4090.sh || echo "[WARN] v26+UDP-GMM+v28 failed or stopped early"

echo "[$(date)] All runs finished. Running post-queue benchmark..."
bash scripts/benchmark_v26_full_queue_local_4090.sh || echo "[WARN] benchmark failed"

echo "[$(date)] Post-queue benchmark finished."
