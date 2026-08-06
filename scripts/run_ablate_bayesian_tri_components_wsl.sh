#!/usr/bin/env bash
# CPU/GPU ablation runner for Bayesian triangulation components.
#
# Smoke run (CPU, <2 min):
#   ./scripts/run_ablate_bayesian_tri_components_wsl.sh --smoke --variant full
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
python -u experiments/ablate_bayesian_tri_components.py "$@"
