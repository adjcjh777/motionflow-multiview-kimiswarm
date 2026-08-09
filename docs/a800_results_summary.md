# A800-D Historical Results Summary

Read-only summary extracted from `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/` on A800-D.

## Best val_MPJPE across completed A800 runs

| val_MPJPE (mm) | Run |
|---------------|-----|
| 17.17 | omniview_fusion_v25_geometry_fusion_full.log |
| 18.31 | omniview_fusion_v25_geometry_fusion_small.log |
| 18.47 | v25_geometry_fusion_small_gpu6_geom1_0.log |
| 18.65 | omniview_fusion_v6_h36m_isab.log |
| 20.06 | omniview_fusion_v11_irls.log |
| 20.16 | omniview_fusion_v10_aleatoric_outlier.log |
| 20.24 | omniview_fusion_v18_deformable_attention.log |
| 20.56 | omniview_fusion_v12_adaptive_multiscale.log |
| 20.89 | omniview_fusion_v18_deformable_attention_fullscale.log |
| 21.54 | omniview_fusion_v29o_hierarchical_n_st_3_a800.log |
| 23.14 | omniview_fusion_v11_irls_fullscale.log |
| 23.52 | omniview_fusion_v10_no_outlier.log |
| 24.88 | omniview_fusion_v4_noskel_mpi.log |
| 25.09 | omniview_fusion_v4_adaptive_hard_mpi.log |
| 25.11 | omniview_fusion_v4_adaptive_mpi.log |
| 25.16 | omniview_fusion_v4_varview_mpi.log |
| 25.16 | omniview_fusion_v4_varview_adaptive_mpi.log |
| 25.90 | omniview_fusion_v31_domain_balanced_a800.log |
| 26.12 | omniview_fusion_v4_skelvec_stable_mpi.log |
| 26.42 | omniview_fusion_v5_mpi.log |
| 26.49 | omniview_fusion_v32_combined_a800.log |
| 26.51 | omniview_fusion_v32_trajectory_consistency_a800.log |
| 26.76 | omniview_fusion_v23_kap_no_ba_gpu6.log |
| 26.97 | omniview_fusion_v31_hierarchical_more_dropout_a800.log |
| 27.43 | omniview_fusion_v32_domain_aware_view_curriculum_a800.log |
| 27.58 | omniview_fusion_v29u_hierarchical_n_heads_2_a800.log |
| 27.75 | omniview_fusion_v32_physical_alignment_a800.log |
| 28.02 | omniview_fusion_v29z_hierarchical_part_layers_2_a800.log |
| 28.12 | omniview_fusion_v29a_hierarchical_only_a800.log |
| 28.41 | omniview_fusion_v31_physical_floor_only_a800.log |
| 28.96 | omniview_fusion_v5_camonly_mpi.log |
| 33.69 | omniview_fusion_v31_geometry_attention_a800.log |
| 37.87 | omniview_fusion_v31_outlier_view_adaptive_a800.log |
| 49.37 | omniview_fusion_v4_skelvec_mpi.log |
| 50.03 | omniview_fusion_v23_kap_no_ba_gpu4.log |
| 56.57 | v6_h36m_debug2.log |
| 58.72 | omniview_fusion_v23b_kap001_no_ba_small.log |
| 82.25 | v6_smoke_test.log |
| 90.28 | omniview_fusion_v29d_tte_physical_only_a800.log |
| 90.35 | omniview_fusion_v29b_hierarchical_tte_a800.log |
| 94.66 | domain_smoke.log |
| 128.27 | omniview_fusion_v21_neural_bundle_adjustment.log |

## Observations

- The strongest completed A800 run is **v25 geometry fusion full at 17.17 mm**, with v25 small at 18.31 mm.
- v32 combined and v32 trajectory consistency are around 26.5 mm, comparable to local v36 (26.42 mm) but far behind v25 full.
- v31 geometry attention (33.69 mm) and outlier view adaptive (37.87 mm) underperform on A800.
- v29 TTE variants fail (~90 mm), consistent with local results.
- Many later variants (v33/v34/v35/v36/v37/v38/v39/v40/v41/v42/v43) are either still running on A800 or queued; their logs are not yet in the top completed list.

## Implications for v44+

1. The v25 geometry-fusion baseline is stronger than many later stacks. We should consider whether the complexity added after v25 is justified.
2. If v36/v42/v43 do not beat v25 full (~17 mm) on A800, we may need to either (a) return to a v25-based stack and add only high-ROI components, or (b) increase model/data scale.
3. WebBridge mixed training and full A800 runs are needed before drawing firm conclusions; local RTX 4090 results may not translate directly.
4. The best A800 run (v25 full) reached its best value at **epoch 1** and then overfit (17.17 mm -> 59.14 mm). This suggests heavy overfitting is a systemic issue. Future runs should prioritize epoch-1 validation and strong regularization/early stopping.
