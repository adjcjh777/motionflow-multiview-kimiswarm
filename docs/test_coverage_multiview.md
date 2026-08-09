# Multi-View Fusion Test Coverage Review

**Scope:** This document inventories which multi-view fusion components in `motionflow_mv/fusion/` and `motionflow_mv/models/` are covered by tests under `tests/`, identifies coverage gaps, and proposes new tests to fill them.

**Methodology:**
- Collected 109 test files from `tests/`.
- Mapped test imports against the 153 modules in `motionflow_mv/fusion/`.
- Cross-checked active toggles in `OmniMultiViewFusionV5` (`motionflow_mv/fusion/omniview_fusion_v5.py`) because that is the current integration point for most fusion features.

---

## 1. Tests That Cover Multi-View Fusion

### Core triangulation and geometry

| Test file | Module(s) covered | What is verified |
|-----------|-------------------|------------------|
| `tests/test_triangulation.py` | `motionflow_mv.fusion.triangulation` | DLT triangulation basic shape/gradients. |
| `tests/test_bayesian_tri_v2_batched_dlt.py` | `motionflow_mv.fusion.triangulation` (`triangulate_dlt_batched_lstsq`) | Batched DLT, weights, gradients, numpy parity. |
| `tests/test_bayesian_tri_v2_batched_dlt_v2.py` | `triangulation`, ray-attention Bayesian tri model | Minimum-2-views, precision-matrix robustness. |
| `tests/test_bayesian_tri_v3.py` | `motionflow_mv.fusion.prototypes.bayesian_tri_v3_model` | SPD joint precision, forward/backward. |
| `tests/test_uncertainty_weighted_triangulation.py` | `motionflow_mv.fusion.uncertainty_weighted_triangulation` | Shape, identity, gradients. |
| `tests/test_multiview_geometry_fusion_v25.py` | `motionflow_mv.fusion.multiview_geometry_fusion_v25` | Forward shape, identity-at-init, gradient flow, view masking, toggle coverage. |
| `tests/test_temporal_geometry_fusion_v26.py` | `motionflow_mv.fusion.temporal_geometry_fusion_v26` | Shape, identity, gradient flow. |
| `tests/test_uncertainty_depth_proposal_v27.py` | `motionflow_mv.fusion.uncertainty_depth_proposal_v27` | Forward/backward, GMM behaviour. |
| `tests/test_uncertainty_aware_triangulation_v33.py` | `motionflow_mv.fusion.uncertainty_aware_triangulation_v33` | Shape, identity, view-mask, gradients. |
| `tests/test_outlier_view_detector.py` | `motionflow_mv.fusion.outlier_view_detector` | Down-weighting behaviour. |
| `tests/test_outlier_view_detector_v33.py` | `motionflow_mv.fusion.outlier_view_detector_v33` | Identity, outlier detection, view masking. |
| `tests/test_ray_conditioned_attention_v33.py` | `motionflow_mv.fusion.ray_conditioned_attention_v33` | Shape, identity-at-init, view-mask. |

### OmniMultiViewFusion integration tests

| Test file | Module(s) covered | What is verified |
|-----------|-------------------|------------------|
| `tests/test_omniview_fusion_v4.py` | `omniview_fusion_v3/v4`, `variable_view_inference` | Shape, gradient, warm-start from v3, all v4 toggles, variable views, entropy loss. |
| `tests/test_omniview_fusion_v5_cross_view_transformer.py` | `omniview_fusion_v5` + `cross_view_transformer_v17` | v17 integration, variable-view mask, default v5. |
| `tests/test_omniview_fusion_v5_deformable_attention.py` | `omniview_fusion_v5` + `deformable_cross_view_attention` | v18 deformable attention, top-k straight-through, variable views. |
| `tests/test_omniview_fusion_v5_robust_dlt.py` | `omniview_fusion_v5` | v8 robust DLT reweight, outlier view, variable views. |
| `tests/test_omniview_fusion_v5_udp_wiring.py` | `omniview_fusion_v5` | v25/v26 geometry fusion + v27 UDP + v28 physical loss end-to-end. |
| `tests/test_eval_omniview_fusion_v2/v3/v4.py` | `omniview_fusion_v2/v3/v4` pipeline | Evaluation / inference paths. |
| `tests/test_train_omniview_fusion_v2_smoke.py` | `omniview_fusion_v2` | Smoke training. |
| `tests/test_train_omniview_fusion_v4_smoke.py` | `omniview_fusion_v4` | Smoke training. |

### Cross-view attention and graph networks

| Test file | Module(s) covered | What is verified |
|-----------|-------------------|------------------|
| `tests/test_cross_view_transformer_v17.py` | `motionflow_mv.fusion.cross_view_transformer_v17` | Forward shape, masking. |
| `tests/test_deformable_cross_view_attention.py` | `motionflow_mv.fusion.deformable_cross_view_attention` | Deformable cross-view attention shape/gradients. |
| `tests/test_cross_view_graph_attention.py` | `motionflow_mv.fusion.prototypes.cross_view_graph_attention`, `graph_joint_relation` | Layer/stack forward, cross-view edge reachability. |
| `tests/test_graph_joint_relation_pp.py` | `graph_joint_relation`, ray-attention graph model | Skeleton graph wiring. |

### Hierarchical / set / temporal encoders

| Test file | Module(s) covered | What is verified |
|-----------|-------------------|------------------|
| `tests/test_hierarchical_multiview_v30.py` | `motionflow_mv.fusion.hierarchical_multiview_v30` | Identity-at-init, view-mask. |
| `tests/test_self_evolving_hierarchical_multiview_v29.py` | `motionflow_mv.fusion.self_evolving_hierarchical_multiview_v29` | Forward, physical loss. |
| `tests/test_multiscale_temporal.py` | `multiscale_temporal_conv_model`, `ray_attention_temporal_model` | Multi-scale temporal shape. |
| `tests/test_temporal_perceiver_v19.py` | `motionflow_mv.fusion.temporal_perceiver_v19` | Forward, mask, gradients. |
| `tests/test_uncertainty_gated_iterative_graph_refinement_v36.py` | `motionflow_mv.fusion.uncertainty_gated_iterative_graph_refinement_v36` | Forward, view-mask, identity. |

### View selection, visibility, and reliability

| Test file | Module(s) covered | What is verified |
|-----------|-------------------|------------------|
| `tests/test_adaptive_view_selector.py` | `motionflow_mv.fusion.adaptive_view_selector` | Forward, budget, bypass, gradients. |
| `tests/test_visibility_gated_fusion_v2.py` | `motionflow_mv.fusion.visibility_gated_fusion_v2` | Shape, mask, gating. |
| `tests/test_attention_entropy_loss.py` | `motionflow_mv.fusion.attention_entropy_loss` | Loss properties, gradients. |
| `tests/test_self_critique_view_reliability_v37.py` | `motionflow_mv.fusion.self_critique_view_reliability_v37` | Shape, range, view-mask. |
| `tests/test_variable_view_inference_hardened.py` | `variable_view_inference`, `omniview_fusion_v2` | Variable view robustness. |
| `tests/test_confidence_resample_dropout.py` | augmentation helpers | Confidence-aware view dropout. |

### Refiners / priors / physical losses

| Test file | Module(s) covered | What is verified |
|-----------|-------------------|------------------|
| `tests/test_skeleton_graph_residual_refiner.py` | `motionflow_mv.fusion.skeleton_graph_residual_refiner` | Shape, identity, gradients. |
| `tests/test_kinematic_chain_graph_refiner.py` | `motionflow_mv.fusion.kinematic_chain_graph_refiner` | Forward, graph structure. |
| `tests/test_kinematic_anthropometric_prior_v22.py` | `motionflow_mv.fusion.kinematic_anthropometric_prior_v22` | Forward, prior losses. |
| `tests/test_smpl_prior_fusion_v22.py` | `motionflow_mv.fusion.smpl_prior_fusion_v22` | Shape, gradients. |
| `tests/test_physical_space_alignment_v28.py` | `motionflow_mv.fusion.physical_space_alignment_v28` | Shape, identity, gradients, floor/bone-temporal losses. |
| `tests/test_physical_loss_warmup.py` | `self_evolving_hierarchical_multiview_v29` physical loss | Warm-up scheduling. |
| `tests/test_rotation_correction.py` | `motionflow_mv.fusion.rotation_correction` | SO(3) residual, bounded rotation. |
| `tests/test_diffusion_pose_refiner_v20.py` | `motionflow_mv.fusion.diffusion_pose_refiner_v20` | Diffusion refiner forward. |
| `tests/test_neural_bundle_adjustment_v21.py` / `..._identity.py` | `motionflow_mv.fusion.neural_bundle_adjustment_v21` | Identity, bundle-adjustment loop. |
| `tests/test_test_time_self_evolution_v27.py` | `motionflow_mv.fusion.test_time_self_evolution_v27` | TTE loop, no NaNs. |
| `tests/test_camera_refinement_v26.py` | `motionflow_mv.fusion.camera_refinement_v26` | Reprojection loss, gate learning. |
| `tests/test_principal_point_correction.py` | `principal_point_correction` | Principal-point residual. |

### Adapter / pipeline / data tests

| Test file | Module(s) covered | What is verified |
|-----------|-------------------|------------------|
| `tests/test_multiview_adapter.py` | `fusion_module.DLTFusion`, `attention_fusion_module.AttentionFusionModule` | Adapter forward, registry. |
| `tests/test_pipeline_multiview_plugin.py` | `pipeline_multiview_plugin` | End-to-end pipeline smoke. |
| `tests/test_pipeline_synthetic.py` | synthetic pipeline | Synthetic multi-view data round-trip. |
| `tests/test_synchronized_multiview_2d_aug.py` | `synchronized_multiview_2d_aug` / `sync_multiview_aug` | 2-D augmentation consistency across views. |
| `tests/test_webbridge_mixed_dataset_v25.py` | data mixing | Domain mixing in the loader. |

---

## 2. Coverage Gaps

The following modules are imported by `OmniMultiViewFusionV5` or are otherwise part of the active multi-view fusion stack, but have **no dedicated unit test**.

### High-priority gaps (actively wired into v5)

| Module | Why it matters | Current coverage |
|--------|----------------|------------------|
| `motionflow_mv.fusion.hierarchical_multiview_v31` (`HierarchicalViewEncoderV31`) | v31 geometry-biased hierarchical encoder; default v5 toggle `use_hierarchical_multiview_v31`. | None. v30 is tested, v31 is not. |
| `motionflow_mv.fusion.hierarchical_multiscale_spatial_pyramid_v33` (`HierarchicalMultiscaleCrossViewSpatialPyramidV33`) | v33 HMSP; default v5 toggle `use_hierarchical_multiscale_spatial_pyramid_v33`. | None. |
| `motionflow_mv.fusion.view_joint_graph_network_v34` (`ViewJointGraphNetworkV34`) | v34 view-joint graph network. | None. |
| `motionflow_mv.fusion.geometry_view_joint_graph_network_v34` (`GeometryViewJointGraphNetworkV34`) | v34 geometry-aware variant. | None. |
| `motionflow_mv.fusion.temporal_view_joint_graph_network_v35` (`TemporalViewJointGraphNetworkV35`) | v35 adds temporal edges. | None. |
| `motionflow_mv.fusion.trajectory_consistency_v32` (`TrajectoryConsistencyV32`) | v32 temporal consistency loss. | None. |
| `motionflow_mv.fusion.camera_view_embedding_v31` (`CameraConditionedViewEmbeddingV31`) | v31 camera-conditioned view embedding; v5 supports `use_camera_view_embedding_v31`. | None. Legacy `camera_conditioned_view_embedding.py` also untested. |
| `motionflow_mv.fusion.skeleton_graph_residual_refiner_v31` (`SkeletonGraphResidualRefinerV31`) | v5 toggle `use_skeleton_residual_v31`. | Only v1 `skeleton_graph_residual_refiner.py` is tested. |
| `motionflow_mv.fusion.perceiver_view_aggregator` (`PerceiverViewAggregator`) | v5 toggle `use_perceiver_aggregator`. | None. |
| `motionflow_mv.fusion.variable_view_set_aggregator` (`VariableViewSetAggregator`) | Variable-view set aggregation. | None. |

### Medium-priority gaps (supporting modules)

| Module | Why it matters | Current coverage |
|--------|----------------|------------------|
| `motionflow_mv.fusion.adaptive_hierarchical_multiscale_fusion` | Early adaptive multi-scale fusion prototype; still present in `motionflow_mv/fusion`. | None. |
| `motionflow_mv.fusion.visibility_gated_fusion.py` (v1) | Predecessor to `visibility_gated_fusion_v2.py`. | Only v2 is tested. |
| `motionflow_mv.fusion.robust_triangulation.py` / `robust_triangulation_baseline.py` / `*_module.py` | Robust triangulation alternatives to the v8 DLT reweight path. | None. |
| `motionflow_mv.fusion.residual_refiner.py` / `residual_refiner_module.py` | Generic residual refiners used by several ray-attention models. | None. |
| `motionflow_mv.fusion.temporal_refiner.py` / `temporal_refiner_module.py` | Temporal refinement helpers. | None. |
| `motionflow_mv.fusion.cross_view_spatial_pyramid.py` | Spatial pyramid cross-view block. | None. |
| `motionflow_mv.fusion.cross_view_visibility_transformer.py` | Cross-view visibility transformer. | None. |
| `motionflow_mv.fusion.epipolar_attention_bias.py` / `epipolar_transformer_bias.py` | Geometry bias primitives used by v31 / v33. | Indirectly exercised only through higher-level tests. |

### Low-priority / prototype gaps

- Many legacy `ray_attention_*_model.py` files (e.g. `ray_attention_v2_model.py`, `ray_attention_temporal_residual_v2_model.py`, etc.) are historical prototypes with no tests. These are largely superseded by `OmniMultiViewFusionV5` and its focused test files.
- `multiperson_association_graph.py`, `semantic_action_conditional_fusion_model.py`, `domain_adaptation_wrapper.py`, `camera_centric_coordinate_transform.py`, `intrinsic_correction.py`, etc. are either prototypes or not on the v5 critical path.

---

## 3. Proposed New Tests

### 3.1 High-priority unit tests

1. **`tests/test_hierarchical_multiview_v31.py`**
   - Instantiate `HierarchicalViewEncoderV31` with/without geometry bias and ray attention.
   - Assert identity-at-init (gate ~0) so regressions in residual gating are caught.
   - Test view masking and gradient flow for `J=17` and `J=28`.
   - Verify that `v31_use_ray_attention=True` still returns the expected shape and finite gradients.

2. **`tests/test_hierarchical_multiscale_spatial_pyramid_v33.py`**
   - Forward pass on `(B, T, V, J, d)` tokens and `(B, T, V, J, 2)` observations.
   - Test each scale individually and the adaptive fusion path.
   - Confirm masked views do not contribute and gradients reach camera/2-D inputs.

3. **`tests/test_view_joint_graph_network_v34.py`**
   - Test `ViewJointGraphNetworkV34` forward, view-mask, identity-at-init, and gradient flow.
   - Include both `J=17` and `J=28` skeletons.

4. **`tests/test_geometry_view_joint_graph_network_v34.py`**
   - Same as above plus geometry inputs (cameras + 2-D points) and verify epipolar/ray bias branches.

5. **`tests/test_temporal_view_joint_graph_network_v35.py`**
   - Test `TemporalViewJointGraphNetworkV35` with `T > 1`.
   - Verify temporal edges change outputs compared to temporal-agnostic v34 when gates are open.

6. **`tests/test_trajectory_consistency_v32.py`**
   - Apply the loss to synthetic trajectories.
   - Verify non-negativity, zero for constant-velocity trajectories (if designed as such), and gradients w.r.t. 3-D pose.

7. **`tests/test_camera_view_embedding_v31.py`**
   - Test `CameraConditionedViewEmbeddingV31` with varying intrinsics/extrinsics.
   - Assert output shape and that different camera rigs produce different embeddings.
   - Verify it is permutation-invariant across views when appropriate.

8. **`tests/test_skeleton_graph_residual_refiner_v31.py`**
   - Forward/backward and identity-at-init for `SkeletonGraphResidualRefinerV31`.
   - Test both skeleton sizes.

9. **`tests/test_perceiver_view_aggregator.py`**
   - Forward shape with variable views.
   - Assert permutation invariance and view-mask handling.

10. **`tests/test_variable_view_set_aggregator.py`**
    - Shape, variable-view mask, and identity behaviour for `VariableViewSetAggregator`.

### 3.2 Integration tests in v5

Add new functions to `tests/test_omniview_fusion_v5_*.py` or create:

- `tests/test_omniview_fusion_v5_hierarchical_v31.py`
  - Enable `use_hierarchical_multiview_v31` and run forward/backward with `v31_use_ray_attention={True,False}`.
- `tests/test_omniview_fusion_v5_hmsp_v33.py`
  - Enable `use_hierarchical_multiscale_spatial_pyramid_v33`.
- `tests/test_omniview_fusion_v5_vjgn_v34.py`
  - Enable `use_view_joint_graph_network_v34`.
- `tests/test_omniview_fusion_v5_tvjgn_v35.py`
  - Enable `use_temporal_view_joint_graph_network_v35`.
- `tests/test_omniview_fusion_v5_trajectory_v32.py`
  - Enable `use_trajectory_consistency_v32` and assert the returned auxiliary loss is finite and non-negative.
- `tests/test_omniview_fusion_v5_perceiver_aggregator.py`
  - Enable `use_perceiver_aggregator`.

### 3.3 Medium-priority unit tests

- `tests/test_robust_triangulation.py` / `test_robust_triangulation_baseline.py`
  - Test that the robust triangulation heads handle outliers and return finite 3-D poses.
- `tests/test_cross_view_spatial_pyramid.py`
  - Forward shape and gradient flow.
- `tests/test_cross_view_visibility_transformer.py`
  - Forward shape and visibility gating.
- `tests/test_residual_refiner.py` / `test_temporal_refiner.py`
  - Smoke tests for the generic residual/temporal refiner modules.
- `tests/test_epipolar_attention_bias.py`
  - Unit tests for `compute_epipolar_distance` and the transformer bias helper with synthetic points and known epipolar geometry.

### 3.4 Regression / end-to-end tests

- Add a single `tests/test_omniview_fusion_v5_all_current_toggles.py` that enables every v5 toggle that currently has no test (v31, v33, v34, v35, v32, perceiver, v31 camera embedding, v31 skeleton residual) and checks forward/backward with view masking. This is intentionally a coarse integration smoke test that catches wiring regressions.

---

## 4. Recommendations

1. **Prioritize v31, v33, v34, v35, and v32** because they are exposed as toggles in `OmniMultiViewFusionV5` and are part of the current A800/RTX 4090 experimental queue.
2. **Mirror the pattern used for v30:** `test_hierarchical_multiview_v30.py` already provides a compact template (identity-at-init, view-mask, parametrized joints). Re-use it for v31, v33, v34, v35.
3. **Keep tests CPU-only and small** (`B=2`, `T<=5`, `d<=64`) to keep the local smoke-test loop fast and avoid GPU allocation on the RTX 4090.
4. **Run `pytest tests/test_*.py -m "not gpu" --collect-only`** after adding tests to confirm no broken imports or naming collisions.
5. **Update this doc** whenever a new fusion toggle is added to `OmniMultiViewFusionV5`.
