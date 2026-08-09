# A800-D Queue Status

## Snapshot

- **Log source:** `outputs/v33_a800_queue.log`
- **Latest status line:** ~line 259 of 286
- **Recorded time:** `Sun Aug  9 08:16:37 2026` (poller restart)
- **State:** GPUs 0-3 still held by VLLM. GPU 6 freed up and the poller launched `v25_geometry_fusion_all_train_baseline`. All-train manifests switched to existing `webbridge_h36m_mpi_mixed_train_val.yaml` because A800 lacks aistpp/3dpw data. The v25 all-train baseline is now running; other priority runs remain queued until GPU memory frees.

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

- GPU 6 freed ~35 GiB; the poller launched the four priority comparison runs on GPU 6.
- `v45_adaptive_geometry_fusion_all_train` remains queued behind the priority runs.
- The log now reports `No GPU with >= 30000 MiB free` again after launching on GPU 6.
