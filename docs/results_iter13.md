# Iter13 Results Snapshot

**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE

## SOTA Comparison (MPI-INF-3DHP S2/Seq1)

| Method | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Raw DLT | 25.21 | 24.08 | 0.990 | 0.999 | 1.000 | 0.832 |
| Robust IRLS | 25.20 | 24.07 | 0.990 | 0.999 | 1.000 | 0.832 |
| Learned PP (ours) | 9.32 | 5.37 | 1.000 | 1.000 | 1.000 | 0.938 |

Script: `experiments/compare_sota_baselines.py`

## Robustness Matrix (PP baseline, 6-axis corruption)

| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|---|---|---|
| clean | 9.32 | 5.37 |
| rot_0.5_deg | 22.62 | 9.10 |
| rot_1.0_deg | 49.06 | 16.88 |
| trans_5mm | 9.88 | 5.50 |
| trans_10mm | 13.14 | 6.00 |
| focal_1pct | 9.54 | 6.12 |
| focal_2pct | 10.49 | 7.01 |
| cxcy_3px | 1791.20 | 463.70 |
| cxcy_5px | 2053.81 | 415.20 |
| distortion_k1_0.01 | 10.45 | 6.94 |
| distortion_k1_0.05 | 18.84 | 17.01 |
| distortion_k1_0.10 | 30.62 | 29.90 |
| view_dropout_0.2 | 13.19 | 6.27 |
| view_dropout_0.4 | 23.15 | 7.69 |
| joint_dropout_0.2 | 14.24 | 12.58 |
| joint_dropout_0.4 | 20.25 | 19.83 |

Script: `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`

## Factorized ST+PP Smoke

| Config | Params | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| d=32, h=64, 500 samples, 5 ep | 90.6 k | 57.68 | 36.68 | 0.460 | 0.928 | 0.999 | 0.616 |

This is a smoke test; a full-capacity run is queued.

## Running / Queued GPU Experiments

1. **PP robust re-train** — in progress (20 epochs, intrinsics curriculum)
2. **Factorized ST+PP full** — queued
3. **SSL pre-training on H36M** — queued
4. **Spatiotemporal PP** — queued

## Notes

- The PP correction head on the 9.32 mm baseline saturates at `max_offset` under real PP perturbation. The robust re-train uses a direct PP loss + curriculum ramp to address this.
- Visibility v2 was deprioritized after training proved CPU-bound on the RTX 4090; it may be revisited on A800-D or after queue items finish.
- A800-D SSH is currently unreachable from this session.
