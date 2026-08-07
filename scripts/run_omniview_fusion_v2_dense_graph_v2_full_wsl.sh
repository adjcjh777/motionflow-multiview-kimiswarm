#!/usr/bin/env bash
# WSL full runner for OmniMultiViewFusionV2 with dense joint attention + graph v2.
# This is the "more complex" multi-view model follow-up to the no-graph ablation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${MF_VENV:-${ROOT}/.venv}
. "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

WARM_START="${WARM_START:-${ROOT}/outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth}"
WARM_START_ARGS=""
if [[ -f "${WARM_START}" ]]; then
  WARM_START_ARGS="--warm_start ${WARM_START} --warm_start_freeze_epochs 5"
fi

# Use all MPI-INF-3DHP train subjects (S1, S3-S8); S2/Seq1 is reserved for validation.
TRAIN_FILES=(
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_04_seq_01_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_04_seq_02_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_05_seq_01_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_05_seq_02_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_06_seq_01_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_06_seq_02_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_07_seq_01_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_07_seq_02_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_08_seq_01_v14_multiview_m.npz"
  "${ROOT}/data/webbridge/mpi_inf_3dhp/s_08_seq_02_v14_multiview_m.npz"
)

python "${ROOT}/experiments/train_omniview_fusion_v2_mpiinf3dhp.py" \
  --train "${TRAIN_FILES[@]}" \
  --val "${ROOT}/data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz" \
  --d 128 \
  --residual_hidden 256 \
  --n_st_layers 3 \
  --graph_num_layers 1 \
  --n_joint_layers 1 \
  --n_heads 4 \
  --epochs 30 \
  --batch_size 8 \
  --train_samples 4000 \
  --val_stride 10 \
  --lr 1e-3 \
  --lr_cosine \
  --lr_warmup_epochs 3 \
  --lr_min 1e-6 \
  --max_grad_norm 1.0 \
  --ema_decay 0.999 \
  --view_dropout_rate 0.1 \
  --min_views 2 \
  --visibility_loss_weight 0.1 \
  --uncertainty_loss_weight 0.05 \
  --temporal_loss_weight 0.02 \
  --bone_loss_weight 0.05 \
  ${WARM_START_ARGS} \
  --output "${ROOT}/outputs/omniview_fusion_v2_d128_dense_graph_v2.pth"
