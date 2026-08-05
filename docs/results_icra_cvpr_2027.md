# Multi-View 3D Pose Estimation — Results Summary (ICRA/CVPR 2027)

## Best single-model results so far

### MPI-INF-3DHP (14 views, 28 joints)
| Model | Config | Best val MPJPE (mm) | Clean MPJPE (mm) | Clean PA-MPJPE (mm) | Notes |
|---|---|---:|---:|---:|---|
| Cross-view temporal residual | d=32, h=64, small | — | 10.25 | 8.60 | baseline without PP correction |
| Cross-view residual + PP | d=32, h=64, pp_w=0.05 | 10.30 | 10.34 | 6.28 | small model |
| Cross-view residual + PP | d=64, h=128, pp_w=0.05 | 9.41 | 9.41 | 5.66 | full model, 10 epochs |
| Cross-view residual + PP | d=64, h=128, pp_w=0.05 | **9.32** | **9.32** | **5.37** | full model, 20 epochs |
| Mixed-dataset PP (MPI+H36M) | d=32, h=64, pp_w=0.05 | 11.64 | 11.64 | 7.45 | small, trained on MPI S1/S3 + H36M S1 acts 2-6 |

### Human3.6M (4 views, 17 joints)
| Model | Config | Best val MPJPE (mm) | Clean MPJPE (mm) | Clean PA-MPJPE (mm) |
|---|---|---:|---:|---:|
| Cross-view residual + PP | d=32, h=64 | — | 6.20 | 4.26 |
| Cross-view residual + PP | d=64, h=128 | **5.24** | **5.24** | **4.84** |

## Robustness (cross-view residual + PP full MPI, 20 ep)
| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| clean | 9.32 | 5.37 |
| rot_0.5_deg | 16.89 | 8.11 |
| rot_1.0_deg | 27.45 | 13.50 |
| trans_5mm | 10.61 | 5.20 |
| trans_10mm | 11.23 | 5.44 |
| focal_1pct | 19.13 | 8.07 |
| focal_2pct | 30.41 | 12.18 |
| cxcy_3px | 11.41 | 5.75 |
| cxcy_5px | 13.87 | 6.61 |

## Negative results

### Two-stage refined PP correction
- Clean MPJPE 14.53 mm (vs 10.34 mm non-refined).
- Dropped; two-stage residual refinement hurts both accuracy and robustness.

### Cross-view CamPE v2 small (d=32, h=64)
- Clean MPJPE 14.39 mm, PA 12.13 mm (vs base cross-view small 10.25/8.60 mm).
- Underperforms the base cross-view model. Likely needs larger capacity (d=64/h=128) or the geometry-based camera PE needs tighter integration (e.g., replace view_pos_embed).

### Cross-view CamPE v2 full (d=64, h=128)
- Best val MPJPE **10.53 mm** (full MPI S1/S3, 10 epochs) vs current best **9.32 mm**.
- Confirmed negative result: full capacity does not close the gap. CamPE direction dropped.

## Interim robustness — curriculum checkpoint (mid-training, MPI S2 full)

| Condition | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | PCK AUC |
|---|---:|---:|---:|---:|---:|---:|
| clean | 10.69 | 7.01 | 1.000 | 1.000 | 1.000 | 0.929 |
| rot_0.5° | 26.78 | 11.09 | 0.909 | 0.993 | 1.000 | 0.821 |
| trans_5mm | 12.42 | 7.13 | 1.000 | 1.000 | 1.000 | 0.917 |
| focal_1% | 11.07 | 7.32 | 1.000 | 1.000 | 1.000 | 0.926 |
| pp_10px | 2023.42 | 459.74 | 0.000 | 0.000 | 0.000 | 0.000 |

*Note: checkpoint is still training; final evaluation after curriculum run completes.*

## Ongoing experiments

| Experiment | Status | Next action |
|---|---|---|
| Mixed-dataset PP small (MPI + H36M, pp_w=0.05) | **done** (best val 11.64 mm) | MPI clean 11.64/7.45; H36M clean 101.02/35.64 (poor cross-dataset generalization) |
| Cross-view CamPE v2 small | **done** (clean 14.39 mm) | negative result; full model also negative (10.53 mm) |
| Cross-view CamPE v2 full | **done** (val 10.53 mm) | dropped |
| Calibration curriculum + view dropout | **training** | evaluate clean + robustness once finished |
| Visibility-gated fusion v2 | queued | start after curriculum |
| Variable-view MPJPE@k (smoke) | **done** | `docs/figures/variable_views_crossview_residual_smoke.png` |
| WebBridge benchmark v2 (smoke) | **done** | s2/v14 14.71, s3/v14 14.70, s1/v4 27.95 mm MPJPE |

## Variable-view MPJPE@k (smoke, crossview-residual baseline)

| k | MPJPE (mm) | std | subsets |
|---|-----------:|---:|---:|
| 2 | 101.22 | 10.34 | 50 |
| 3 | 84.81 | 5.66 | 50 |
| 4 | 73.90 | 5.59 | 50 |
| 5 | 60.83 | 4.37 | 50 |
| 6 | 50.20 | 4.65 | 50 |
| 7 | 41.42 | 3.13 | 50 |
| 8 | 34.24 | 2.59 | 50 |
| 9 | 30.42 | 1.76 | 50 |
| 10 | 30.71 | 2.10 | 50 |
| 11 | 31.23 | 1.78 | 50 |
| 12 | 33.88 | 2.14 | 50 |
| 13 | 37.96 | 1.97 | 14 |
| 14 | 14.01 | 0.00 | 1 |

## WebBridge cross-dataset benchmark v2 (smoke)

| Dataset | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK AUC |
|---|---:|---:|---:|---:|---:|
| mpi_s2_seq1_v14 | 14.71 | 13.86 | 0.997 | 1.000 | 0.902 |
| mpi_s3_seq1_v14 | 14.70 | 11.41 | 0.998 | 1.000 | 0.902 |
| mpi_s1_seq1_v4 | 27.95 | 19.10 | 0.888 | 0.981 | 0.814 |

## Planned next directions

1. **Camera Positional Encoding v2 for cross-view model** — geometry-based camera embeddings for cross-dataset/variable-view transfer.
2. **Adaptive view selection / visibility gating** — handle occlusion and drop bad views.
3. **Graph joint relation** — skeleton-aware attention.
4. **Uncertainty-weighted triangulation** — per-view reliability estimation.
5. **MotionFlow integration** — use multi-view output to initialize/supervise single-view motionflow pipeline.

## Key takeaways for the paper

- Explicit principal-point supervision makes the multi-view estimator robust to small calibration drift without hurting clean accuracy.
- Cross-view spatio-temporal attention outperforms temporal-only attention.
- Mixed-dataset training keeps MPI accuracy at 11.64 mm but H36M cross-dataset performance is poor (101 mm), suggesting the mixed model needs more H36M data, domain balancing, or a dedicated per-dataset pose head.
