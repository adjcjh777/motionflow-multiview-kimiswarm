#!/usr/bin/env bash
# Evaluate the v51 DAE smoke checkpoint with the standard MPJPE@k protocol.
# Assumes the smoke checkpoint at outputs/v51_dae_smoke_local_4090.pth exists.
set -euo pipefail

CHECKPOINT="outputs/v51_dae_smoke_local_4090.pth"
MANIFEST="configs/splits/webbridge_h36m_mpi_mixed_val_for_eval.txt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    echo "Run scripts/run_v51_domain_agnostic_ensemble_smoke_local_4090.sh first."
    exit 1
fi

python scripts/run_mpjpe_at_k_benchmark.py \
    --checkpoint "$CHECKPOINT" \
    --config "outputs/v51_dae_smoke_local_4090.config.json" \
    --dataset_manifest "$MANIFEST" \
    --k_values 2 3 4 14 \
    --clip_len 9 \
    --output_dir outputs/mpjpe_at_k_v51_dae \
    --device cuda
