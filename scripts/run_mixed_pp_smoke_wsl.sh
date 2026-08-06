#!/usr/bin/env bash
# CPU/GPU smoke test for mixed-dataset principal-point correction training.
# Runs a single tiny epoch with the --smoke flag to verify the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

MPI_TRAIN=(
  data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz
)
H36M_TRAIN=(
  data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz
)

python -u experiments/train_mixed_dataset_principal_point.py \
  --mpi_train "${MPI_TRAIN[@]}" \
  --h36m_train "${H36M_TRAIN[@]}" \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --val_dataset mpi \
  --balance_datasets \
  --smoke \
  --output outputs/mixed_pp_smoke.pth \
  "$@"
