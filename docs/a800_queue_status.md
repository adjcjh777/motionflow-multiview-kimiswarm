# A800-D Queue Status

## Snapshot

- **Log source:** `outputs/v33_a800_queue.log`
- **Latest status line:** ~line 259 of 286
- **Recorded time:** `Sun Aug  9 08:16:37 2026` (poller restart)
- **State:** GPUs 4-7 are at 100% utilization running MotionFlow jobs. GPUs 0-3 are idle in SM but hold ~76 GiB memory each (VLLM::Worker_TP processes), leaving no A800 GPU with >= 30 GiB free. The poller is stalled until one of the MotionFlow jobs on GPUs 4-7 finishes and frees memory.

## Currently Running Jobs (20 total)

| # | Run Name |
|---|----------|
| 1 | `v31_physical_floor_only` |
| 2 | `v31_top5_v31_outlier_view_adaptive` |
| 3 | `v32_combined` |
| 4 | `v32_domain_aware_view_curriculum` |
| 5 | `v32_physical_alignment` |
| 6 | `v32_ray_attention` |
| 7 | `v32_trajectory_consistency_refiner` |
| 8 | `v33_combined_all_three` |
| 9 | `v33_combined_all_three_fixed` |
| 10 | `v33_combined_all_three_plus_hmsp` |
| 11 | `v33_hierarchical_multiscale_spatial_pyramid` |
| 12 | `v33_outlier_view_rejection` |
| 13 | `v33_ray_conditioned_attention` |
| 14 | `v33_uncertainty_aware_triangulation` |
| 15 | `v34_geometry_view_joint_graph_network` |
| 16 | `v34_geometry_view_joint_graph_network_dropout_0_1` |
| 17 | `v34_geometry_view_joint_graph_network_n_layers_1` |
| 18 | `v34_hmsp_geometry_vjgn_stack` |
| 19 | `v34_hmsp_vjgn_stack` |
| 20 | `v34_view_joint_graph_network` |

## Queued Jobs (18 total)

| # | Run Name |
|---|----------|
| 1 | `v25_geometry_fusion_all_train_baseline` |
| 2 | `v25_geometry_fusion_all_train_plus_physical_domain` |
| 3 | `v42_v36_physical_domain_no_v37` |
| 4 | `v43_adaptive_node_residual_on_v42` |
| 5 | `v43_adaptive_node_residual_scaled` |
| 6 | `v43_adaptive_node_residual_all_train` |
| 7 | `v34_geometry_vjgn_combined_fixed_max` |
| 8 | `v35_temporal_vjgn_on_v34_vjgn` |
| 9 | `v35_temporal_vjgn_on_v34_geometry_vjgn` |
| 10 | `v36_ugigr_on_v34_vjgn` |
| 11 | `v36_ugigr_on_v35_tvjgn` |
| 12 | `v36_ugigr_on_v34_hmsp_geometry_vjgn` |
| 13 | `v36_ugigr_n_iters_1_on_v34_vjgn` |
| 14 | `v37_scvr_on_v36_ugigr` |
| 15 | `v38_expanded_data_scvr` |
| 16 | `v39_rcgr_on_v38_scvr` |
| 17 | `v40_skeleton_physical_loss_on_v39_rcgr` |
| 18 | `v41_domain_weighted_loss_on_v40` |

## Notes

- GPUs 4, 5, 6, and 7 are occupied; no launches have completed since the latest `Already-running runs names` dump.
- The remaining `v31_top5_*` runs (e.g., `v31_domain_balanced`, `v31_hierarchical_more_dropout`, `v31_geometry_attention`) appear to have finished, while `v31_physical_floor_only` and `v31_top5_v31_outlier_view_adaptive` are still active.
- Recent additions to the queue include `v42_*`, `v43_*`, and `v25_geometry_fusion_all_train_*` baselines.
- The log ends with repeated `No GPU with >= 30000 MiB free; sleeping 60s`, so the queue is stalled until one of the running jobs frees a GPU.
