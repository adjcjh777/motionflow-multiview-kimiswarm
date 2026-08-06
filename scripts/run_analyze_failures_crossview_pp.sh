#!/usr/bin/env bash
# CPU smoke failure analysis for the cross-view PP + residual model.
# Override DATASET / CHECKPOINT to run on other artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:-data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz}
CHECKPOINT=${2:-outputs/crossview_pp_smoke.pth}

KMP_DUPLICATE_LIB_OK=TRUE \
python -u experiments/analyze_failures_crossview_pp.py \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --clip_len 13 --d 32 --n_st_layers 2 --residual_hidden 64 \
  --batch_size 8 \
  --device cpu \
  --out_dir outputs/failure_analysis_crossview_pp_smoke \
  --report_dir docs/swarm_iter_next/failure_analysis_crossview_pp_smoke
