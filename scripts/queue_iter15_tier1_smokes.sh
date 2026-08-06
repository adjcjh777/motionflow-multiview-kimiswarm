#!/usr/bin/env bash
# Queue Tier-1 iter15 smoke experiments on RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[queue] starting Tier-1 iter15 smokes"

echo "[queue] 1/3 Gaussian-splatting pose regularizer smoke"
bash scripts/run_splat_pp_smoke_wsl.sh

echo "[queue] 2/3 Kinematic-chain graph refiner smoke"
bash scripts/run_kinematic_chain_pp_smoke_wsl.sh

echo "[queue] 3/3 Cross-view contrastive pose representation smoke"
bash scripts/run_crossview_contrast_pp_smoke_wsl.sh

echo "[queue] all Tier-1 iter15 smokes completed"
