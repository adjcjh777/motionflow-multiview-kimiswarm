#!/usr/bin/env bash
# WSL smoke runner for OmniMultiViewFusionV2 (1 epoch, small d).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${MF_VENV:-${ROOT}/.venv}
. "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

python "${ROOT}/experiments/train_omniview_fusion_v2_mpiinf3dhp.py" \
  --train "${ROOT}/data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz" \
          "${ROOT}/data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz" \
          "${ROOT}/data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz" \
          "${ROOT}/data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz" \
  --val "${ROOT}/data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz" \
  --smoke \
  --d 32 \
  --residual_hidden 64 \
  --n_st_layers 1 \
  --graph_num_layers 1 \
  --epochs 1 \
  --batch_size 2 \
  --train_samples 128 \
  --val_stride 50 \
  --lr 1e-3 \
  --max_grad_norm 1.0 \
  --visibility_loss_weight 0.1 \
  --uncertainty_loss_weight 0.05 \
  --output "${ROOT}/outputs/omniview_fusion_v2_smoke.pth"
