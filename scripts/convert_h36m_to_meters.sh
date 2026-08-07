#!/usr/bin/env bash
# Convert WebBridge H36M .npz files from mm to meters.
# Train subjects live in data/webbridge/h36m; val/test subjects are corrected
# versions in data/webbridge/h36m_corrected. Both are scaled by 1000 to meters.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

mkdir -p data/webbridge/h36m_meters

for f in data/webbridge/h36m/s_01_acts_*.npz data/webbridge/h36m_corrected/s_*_acts_*.npz; do
  [ -e "$f" ] || continue
  out="data/webbridge/h36m_meters/$(basename "$f" .npz)_m.npz"
  python -u experiments/convert_npz_to_meters.py --input "$f" --scale 1000 --output "$out"
done
