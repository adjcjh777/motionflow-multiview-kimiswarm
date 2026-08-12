# MPI-INF-3DHP full benchmark protocol

I investigated the MPI-INF-3DHP benchmarking setup and wrote the report to:

**`docs/swarm_iter7/mpi_inf_3dhp_full_benchmark_protocol.md`**

Summary of findings:

- **Current state:** The repo can already convert and evaluate MPI-INF-3DHP sequences (`motionflow_mv/data/webbridge_loader.py:347–413`, `experiments/batch_convert_mpiinf3dhp_v1.py`), but only S1/S2/S3 sequences are present, and every script uses the small smoke split **train S1 → val S2/Seq1** (best 11.17 mm).
- **Gap:** The project is not running the standard full cross-subject benchmark (train S1–S5, test S6–S8), and metrics lack root-relative MPJPE, which is the literature standard.
- **Concrete next step:** Download/convert S4–S8, add a root-relative MPJPE helper to `motionflow_mv/eval/metrics.py` (pelvis index 0), and create `experiments/run_mpiinf3dhp_full_benchmark.py` to train on S1–S5 and evaluate on all S6–S8 test sequences.
- **Success metric:** A per-sequence/per-subject results table with mean test MPJPE/PA-MPJPE/PCK/AUC over S6–S8; target mean test MPJPE ≤ 15 mm.
- **Risks:** ~8 GB of generated `.npz`s (do not commit), GPU contention on the local RTX 4090 / read-only A800-D, and the need to verify the pelvis joint index.