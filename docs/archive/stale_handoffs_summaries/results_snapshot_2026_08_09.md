# Results Snapshot (2026-08-09)

This snapshot captures the best epoch-1 validation MPJPE for local RTX 4090 and A800 runs as of 2026-08-09. It is intended to guide the v42/v43/v44 decision process.

## Local RTX 4090 (best epoch-1)

| Rank | Run | val_MPJPE (mm) | Notes |
|---|---|---|---|
| 1 | v2_d128_no_graph | 24.71 | Local best; simpler architecture |
| 2 | v2_d128_dense_graph_v2 | 25.19 | |
| 3 | v34_hmsp_geometry_vjgn_stack | 25.50 | |
| 4 | v33_combined | 25.78 | |
| 5 | **v42_v36_physical_domain** | **26.16** | Currently running; d=64, old manifest |
| 6 | v36_ugigr | 26.42 | |
| 7 | v37_scvr | 26.94 | |
| 8 | v35_tvjgn | 27.08 | |
| 9 | v34_vjgn | 27.17 | |
| 10 | v33_hmsp | 27.32 | |

## A800 (best epoch-1, historical + currently running)

| Rank | Run | val_MPJPE (mm) | Notes |
|---|---|---|---|
| 1 | **v25_geometry_fusion_full** | **17.17** | Best overall; strong baseline on GPU4 |
| 2 | v25_geometry_fusion_small | 18.31 | |
| 3 | v11_irls | 20.06 | |
| 4 | v10_aleatoric_outlier | 20.16 | |
| 5 | v18_deformable_attention | 20.24 | |
| 6 | v12_adaptive_multiscale | 20.56 | |
| 7 | v18_deformable_attention_fullscale | 20.89 | |
| 8 | v29o_hierarchical_n_st_3 | 21.54 | |
| 9 | v32_combined | 26.49 | v31-v34 complex stack |
| 10 | v32_trajectory_consistency | 26.51 | |
| 11 | v32_ray_attention | 26.58 | |
| 12 | v31_hierarchical_more_dropout | 26.97 | |
| 13 | v33_ray_conditioned_attention | 26.85 | |
| 14 | v33_uncertainty_aware_triangulation | 27.57 | |
| 15 | v32_physical_alignment | 27.75 | |
| 16 | v31_physical_floor_only | 28.41 | |
| 17 | v29a_hierarchical_only | 28.12 | |
| 18 | v33_outlier_view_rejection | 30.57 | |
| 19 | v31_geometry_attention | 33.69 | |
| 20 | v31_outlier_view_adaptive | 37.87 | |

## Key observations

1. **v25 is the strongest baseline.** On A800, v25 geometry fusion reaches 17.17 mm, far below v31-v34 (26–37 mm).
2. **Complexity has not paid off yet.** All v31-v34 variants are worse than v25 and v18.
3. **v42 local is promising but not exceptional.** At 26.16 mm (d=64, old manifest), it ranks behind several simpler local baselines.
4. **A800 v42/v43 results are pending.** The priority queue (`v25 all-train`, `v25+physical+domain`, `v42`, `v43`) will determine the v44 direction.

## Pending experiments

- `v25_geometry_fusion_all_train_baseline`
- `v25_geometry_fusion_all_train_plus_physical_domain`
- `v42_v36_physical_domain_no_v37`
- `v43_adaptive_node_residual_on_v42`
- `v43_adaptive_node_residual_scaled`
- `v43_adaptive_node_residual_all_train`

These are queued on A800 and will launch as GPUs free up.

## Decision implication

Per `docs/v43_decision_criteria.md`:
- If any v42/v43 A800 run beats v25's ~17 mm, continue refining the complex stack.
- If v25 remains best, pivot v44 to a v25-based architecture with selective additions (physical loss, domain weights, etc.).
