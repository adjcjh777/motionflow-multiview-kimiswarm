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

## Ongoing experiments

| Experiment | Status | Next action |
|---|---|---|
| Mixed-dataset PP small (MPI + H36M, pp_w=0.05) | **done** (best val 11.64 mm) | MPI clean 11.64/7.45; H36M clean 101.02/35.64 (poor cross-dataset generalization) |
| Cross-view CamPE v2 small | **done** (clean 14.39 mm) | negative result; try full model or drop |
| Adaptive view selection small | training | evaluate once finished |
| Variable-view inference benchmark | deferred (GPU contention) | re-run after GPU is free |

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
