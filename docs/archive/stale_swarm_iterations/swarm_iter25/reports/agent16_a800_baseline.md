# Agent-16 Analysis: A800 Read-Only Baseline for v48 Domain Generalization

**Scope:** Read A800-D results read-only and update the baseline table that v48 will be compared against. No source files were modified.

**Date:** 2026-08-09

**Sources:**
- `ssh a800-D` read-only queries of `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/docs/`
- `ssh a800-D` read-only queries of `outputs/` and `tmux ls`
- Local `outputs/v46_svg_smoke_local_4090.*` and `outputs/v47_temporal_svg_smoke_local_4090.*`
- Local `docs/proposals/v48_domain_generalization.md`
- Local `docs/swarm_iter25_action_plan.md`

---

## 1. Executive Summary

For v48 domain generalization, the **only A800 baseline that currently exists is the pre-v46 stack**. Neither v46 Sparse-View Generalization nor v47 Temporal Aggregation have completed an A800 run yet, and the local v47 smoke log was truncated at step 200. Therefore the v48 A800 baseline is:

| Rank | Run | Best val_MPJPE (mm) | Hardware | Status |
|------|-----|---------------------|----------|--------|
| 1 | v25 geometry fusion full | **17.17** | A800-D | Completed (best known) |
| 2 | v25 geometry fusion small | 18.31 | A800-D | Completed |
| 3 | v18 deformable attention | 20.24 | A800-D | Completed |
| 4 | v29o hierarchical (n_st=3) | 21.54 | A800-D | Completed |
| 5 | v32 combined | 26.49 | A800-D | Completed |
| 6 | v32 trajectory consistency | 26.51 | A800-D | Completed |
| 7 | v33 ray-conditioned attention | 26.85 | A800-D | Completed |
| 8 | v31 hierarchical more dropout | 26.97 | A800-D | Completed |
| 9 | v32 physical alignment | 27.75 | A800-D | Completed |
| 10 | v33 uncertainty-aware triangulation | 27.57 | A800-D | Completed |

**Key takeaway for v48:** The strongest published A800 result is still **v25 geometry fusion full at 17.17 mm**. Any v48 full-run on A800 should report against this number, not only against v46/v47, until v46/v47 A800 numbers land.

---

## 2. Read-Only A800 Status (2026-08-09)

### 2.1 Currently running tmux sessions on A800-D

```
v25_geometry_fusion_all_train_baseline_gpu6
v25_geometry_fusion_full_gpu4
v25_geometry_fusion_small_gpu6_geom1_0
v25_geometry_fusion_small_gpu7
v31_physical_floor_only_gpu5
v31_top5_v31_outlier_view_adaptive_gpu7
v32_combined_gpu7
v32_domain_aware_view_curriculum_gpu4
v32_physical_alignment_gpu4
v32_ray_attention_gpu5
v32_trajectory_consistency_refiner_gpu6
v33_combined_all_three_gpu4
v33_combined_all_three_plus_hmsp_gpu5
v33_hierarchical_multiscale_spatial_pyramid_gpu7
v33_outlier_view_rejection_gpu4
v33_ray_conditioned_attention_gpu6
v33_uncertainty_aware_triangulation_gpu6
v34_geometry_view_joint_graph_network_dropout_0_1_gpu6
v34_geometry_view_joint_graph_network_gpu7
v34_geometry_view_joint_graph_network_n_layers_1_gpu5
v34_hmsp_geometry_vjgn_stack_gpu4
v34_hmsp_vjgn_stack_gpu5
v34_view_joint_graph_network_gpu4
```

**Observation:** No `v46_*` or `v47_*` sessions are active on A800-D. The v46/v47 A800 queue entries have not started yet, so A800 cannot yet provide a v46/v47 baseline for v48.

### 2.2 A800 historical best val_MPJPE (from `docs/a800_results_summary.md`)

| val_MPJPE (mm) | Run |
|---------------|-----|
| 17.17 | omniview_fusion_v25_geometry_fusion_full.log |
| 18.31 | omniview_fusion_v25_geometry_fusion_small.log |
| 18.47 | v25_geometry_fusion_small_gpu6_geom1_0.log |
| 20.24 | omniview_fusion_v18_deformable_attention.log |
| 20.56 | omniview_fusion_v12_adaptive_multiscale.log |
| 20.89 | omniview_fusion_v18_deformable_attention_fullscale.log |
| 21.54 | omniview_fusion_v29o_hierarchical_n_st_3_a800.log |
| 23.14 | omniview_fusion_v11_irls_fullscale.log |
| 26.42 | omniview_fusion_v36_ugigr_a800.log (local: 26.42) |
| 26.49 | omniview_fusion_v32_combined_a800.log |
| 26.51 | omniview_fusion_v32_trajectory_consistency_a800.log |
| 26.76 | omniview_fusion_v23_kap_no_ba_gpu6.log |
| 26.97 | omniview_fusion_v31_hierarchical_more_dropout_a800.log |
| 27.43 | omniview_fusion_v32_domain_aware_view_curriculum_a800.log |
| 27.57 | v33 uncertainty-aware triangulation |
| 27.75 | omniview_fusion_v32_physical_alignment_a800.log |
| 28.02 | omniview_fusion_v29z_hierarchical_part_layers_2_a800.log |
| 28.41 | omniview_fusion_v31_physical_floor_only_a800.log |
| 30.57 | v33 outlier-view rejection |
| 33.69 | v31 geometry attention |
| 37.87 | v31 outlier view adaptive |

---

## 3. Updated v48 Baseline Table

The table below is the recommended baseline that v48 experiments should reference. It merges the A800 historical best with the pending v46/v47 A800 status.

| Variant | Best A800 val_MPJPE (mm) | Best Local RTX 4090 val_MPJPE (mm) | A800 Status | v48 Relevance |
|---------|--------------------------|-----------------------------------|-------------|---------------|
| v25 geometry fusion full | **17.17** | — | Completed | Absolute baseline; v48 must not regress this on H36M/MPI/AIST full views |
| v25 geometry fusion small | **18.31** | — | Completed | Secondary v25 baseline |
| v18 deformable attention | 20.24 | — | Completed | Proven strong simple stack |
| v29o hierarchical | 21.54 | — | Completed | Hierarchical encoder only |
| v32 combined | 26.49 | — | Completed | First complex stack to watch |
| v32 trajectory consistency | 26.51 | — | Completed | Temporal consistency baseline |
| v33 ray-conditioned attention | 26.85 | — | Completed | Geometry-biased cross-view attention |
| v31 hierarchical more dropout | 26.97 | — | Completed | Regularized hierarchical baseline |
| v33 uncertainty-aware triangulation | 27.57 | — | Completed | Precision-weighted triangulation |
| v42 v36+physical+domain | — | 26.16 | Not run on A800 | Physical + domain-weight loss stack |
| v36 UGIGR | — | 26.42 | Not run on A800 | Uncertainty-gated iterative graph refinement |
| v37 SCVR | — | 26.94 | Not run on A800 | Self-critique view reliability |
| v35 TVJGN | — | 27.08 | Not run on A800 | Temporal view-joint graph |
| v46-SVG | — | smoke only | Queued / not started | Sparse-view generalization; predecessor of v48 |
| v47-temporal | — | smoke truncated at step 200 | Queued / not started | Temporal aggregation; predecessor of v48 |

### Notes on the table

- **v25 is the only A800 baseline with a sub-20 mm val_MPJPE.** All v31–v43 complex stacks are 26+ mm on A800, suggesting the extra complexity has not yet beaten v25.
- **v46/v47 have no A800 results yet.** The local v46 smoke reached step 3350 with training loss decreasing from ~20.9 to ~5.9, but no validation MPJPE was logged in the local smoke file. The local v47 smoke was interrupted at step 200.
- **v48 must therefore be evaluated against v25 for A800 comparisons** until v46/v47 full runs complete.

---

## 4. Variable-View MPJPE@k Results (A800 Curriculum)

From `docs/results_variable_views_curriculum.md` (A800-D variable-view curriculum, mean ± std over 5 subsets):

| k | mean MPJPE (mm) | std (mm) |
|---|-----------------|----------|
| 2 | 79.55 | 34.83 |
| 3 | 39.98 | 9.07 |
| 4 | 35.23 | 12.80 |
| 5 | 30.22 | 8.91 |
| 6 | 24.11 | 5.48 |
| 7 | 23.22 | 8.43 |
| 8 | 21.47 | 3.27 |
| 9 | 19.75 | 5.54 |
| 10 | 17.06 | 1.19 |
| 11 | 12.50 | 1.42 |
| 12 | 12.22 | 1.99 |
| 13 | 10.47 | 0.44 |
| 14 | 9.47 | 0.00 |

**Implication for v48:** v48's domain generalization will be tested with variable view counts. A reasonable v48 target is to remain within ~1.5× of these per-k numbers on H36M/MPI/AIST val, while improving the 3DPW actual monocular (k=1) number.

---

## 5. Cross-Dataset Transfer Baseline

From `docs/results_cross_dataset.md` (H36M-trained model zero-shot on Campus/Shelf):

| Dataset | Views | drop=0.0, noise=0.0 | drop=0.4, noise=5.0 |
|---------|-------|---------------------|---------------------|
| Campus Seq1 | 3 | 0.738 m | 0.645 m |
| Shelf Seq1 | 5 | 0.083 m | 0.074 m |

**Implication for v48:** The current model already transfers to different camera rigs, but the gap between 3-view (Campus) and 5-view (Shelf) is large. v48's domain-invariant wrapper should aim to narrow this gap, especially for the moving-camera 3DPW actual case.

---

## 6. Local Smoke Findings (v46 / v47)

### v46 SVG smoke (`outputs/v46_svg_smoke_local_4090.log`)
- Config: d=64, n_st_layers=2, batch=4, clip_len=9, train_samples=500, epochs=2
- Flags: `use_v45_adaptive_geometry_fusion=true`, `use_v46_sparse_view_generalization=true`, `v46_svg_view_dropout_prob=0.3`, `v46_svg_min_views=2`, `v46_svg_hidden=64`
- Model params: 897,977
- Training loss decreased monotonically from 20.89 at step 50 to 5.89 at step 3350
- **No validation MPJPE reported in the log**; the smoke was run for stability only

### v47 temporal smoke (`outputs/v47_temporal_svg_smoke_local_4090.log`)
- Config: d=64, n_st_layers=2, batch=4, clip_len=9, train_samples=500, epochs=2
- Flags: `use_v46_sparse_view_generalization=true` + `use_v47_temporal_aggregation=true`
- Model params: 1,015,933
- Training loss: 20.71 → 16.55 → 13.90 → 12.07
- **Log truncated at step 200**; no validation numbers available

---

## 7. Implications for v48

1. **A800 baseline is v25, not v46/v47.** Until v46/v47 complete on A800, v48 should be benchmarked primarily against the v25 geometry fusion full result (17.17 mm) for full-view H36M/MPI/AIST, and against the variable-view curriculum table for sparse-view performance.

2. **3DPW actual is unmeasured on A800.** No A800 run has yet used the 3DPW actual-mode loader. The v48 A800 full run will be the first to report `MPJPE@1` on real 3DPW moving-camera sequences.

3. **Local smoke is ahead of A800.** v46 and v47 local smokes exist but lack complete validation numbers. v48 should first complete a stable local smoke that reports per-domain val_MPJPE (H36M/MPI/AIST/3DPW pseudo/3DPW actual) before queuing A800.

4. **v48 target thresholds.** Based on the proposal, v48 should:
   - Match or improve v25 full-view val_MPJPE (≤ 17.17 mm on H36M/MPI/AIST full views).
   - Reduce the 3DPW↔studio gap by ≥ 20% relative to whatever v47 A800 baseline eventually lands.
   - Keep domain discriminator accuracy near chance (0.45–0.55).

---

## 8. Open Gaps / Action Items

| Gap | Owner | Next Step |
|-----|-------|-----------|
| No v46/v47 A800 results | Agent-12 (QUEUE) / orchestrator | Wait for A800 priority queue (v25/v42/v43) to finish, then launch v46/v47 A800 runs |
| No 3DPW actual-mode A800 baseline | Agent-04 (IMPLEMENT), Agent-11 (EVAL) | Implement loader and eval path; first 3DPW actual result will come from v48 smoke/full run |
| No per-domain val_MPJPE in local smokes | Agent-11 (EVAL) | Extend `eval_variable_views.py` to report per-domain MPJPE@k |
| v48 A800 queue entry missing | Agent-12 (QUEUE) | Add v48 full-run entry to `scripts/launch_v33_a800_queue.py` once smoke passes |

---

## 9. Recommendation

**For v48 reporting, use the following baseline statement:**

> As of 2026-08-09, the strongest completed A800 baseline is **v25 geometry fusion full at 17.17 mm val_MPJPE**. The v46 Sparse-View Generalization and v47 Temporal Aggregation modules have only local smoke runs (no A800 validation numbers), so v48 will be compared against v25 for full-view accuracy and against the A800 variable-view curriculum for sparse-view accuracy until v46/v47 A800 results become available.
