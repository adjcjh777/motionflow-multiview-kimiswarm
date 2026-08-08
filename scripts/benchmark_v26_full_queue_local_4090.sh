#!/usr/bin/env bash
# Run the full v5 benchmark on all v26 full-queue checkpoints.
set -euo pipefail

H36M="data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz"
MPI="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"

for variant in v26_udp v26_udp_gmm v26_udp_v28 v26_udp_gmm_v28; do
    CKPT="outputs/omniview_fusion_${variant}_full_local_4090.pth"
    OUT="outputs/benchmark_${variant}_full_local_4090.json"
    CSV="outputs/benchmark_${variant}_full_local_4090.csv"

    if [[ ! -f "$CKPT" ]]; then
        echo "[SKIP] $variant: checkpoint not found ($CKPT)"
        continue
    fi

    echo "[$(date)] Benchmarking $variant ..."
    python scripts/run_full_v5_benchmark.py \
        --checkpoint "$CKPT" \
        --h36m "$H36M" \
        --mpi "$MPI" \
        --out "$OUT" \
        --csv "$CSV" \
        || echo "[WARN] $variant benchmark failed"

    echo "[$(date)] Benchmarking $variant with TTE ..."
    python scripts/run_full_v5_benchmark.py \
        --checkpoint "$CKPT" \
        --h36m "$H36M" \
        --mpi "$MPI" \
        --out "${OUT%.json}_tte.json" \
        --csv "${CSV%.csv}_tte.csv" \
        --tte \
        || echo "[WARN] $variant TTE benchmark failed"
done

echo "[$(date)] All benchmarks finished."
