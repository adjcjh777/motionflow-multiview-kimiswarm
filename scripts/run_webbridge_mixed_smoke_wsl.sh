#!/usr/bin/env bash
# WebBridge mixed-dataset loader smoke test.
# Runs one CPU epoch of the PP anchor on H36M + MPI-INF-3DHP + AIST++.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/outputs/webbridge_mixed_smoke.log"
mkdir -p "$(dirname "$LOG")"

python "${ROOT}/experiments/smoke_webbridge_mixed.py" \
  --train_paths \
    "${ROOT}/data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz" \
    "${ROOT}/data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz" \
    "${ROOT}/data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz" \
  --train_names h36m mpi aist \
  --val_paths \
    "${ROOT}/data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz" \
    "${ROOT}/data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz" \
  --val_names h36m mpi \
  --clip_len 9 \
  --d 32 \
  --n_st_layers 1 \
  --residual_hidden 64 \
  --batch_size 2 \
  --train_samples 20 \
  --epochs 1 | tee "$LOG"
