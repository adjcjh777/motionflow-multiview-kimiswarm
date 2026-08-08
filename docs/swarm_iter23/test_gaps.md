# Multi-view fusion test gaps (swarm iter 23)

_Generated: 2026-08-08_

## Scope

This review covers the `tests/` directory and the fusion modules under `motionflow_mv/fusion/` for the MotionFlow-MultiView multi-view pose-estimation pipeline.

## Summary statistics

- Fusion modules in `motionflow_mv/fusion/`: **137**
- Test files in `tests/`: **88**
- Fusion modules with at least one direct test: **61**
- Fusion modules with no direct test: **76**

## Well-covered areas

| Category | Representative modules | Tests |
|---|---|---|
| Triangulation | `triangulation`, `uncertainty_weighted_triangulation` | `test_triangulation.py`, `test_uncertainty_weighted_triangulation.py` |
| Attention fusion | `attention`, `attention_model`, `attention_fusion_module`, `attention_entropy_loss` | `test_attention.py`, `test_attention_entropy_loss.py` |
| Omniview fusion | `omniview_fusion_v2`, `omniview_fusion_v4`, `omniview_fusion_v5` | `test_omniview_fusion_v4.py`, `test_omniview_fusion_v5_*.py`, `test_eval_omniview_fusion_v*.py` |
| Ray-attention base | `ray_attention_model`, `ray_attention_module`, `ray_attention_temporal_model`, `ray_attention_temporal_crossview_model` | `test_ray_attention.py`, `test_ray_attention_temporal.py`, `test_ray_attention_temporal_crossview.py` |
| Principal-point / residual ray attention | `ray_attention_temporal_crossview_residual_principal_point_model` | `test_iter14_models_train_step.py`, `test_principal_point_correction.py` |
| Cross-view graph attention | `graph_joint_relation`, `prototypes/cross_view_graph_attention` | `test_cross_view_graph_attention.py`, `test_graph_joint_relation_pp.py` |
| Kinematic anthropometric prior (KAP) | `kinematic_anthropometric_prior_v22` | `test_kinematic_anthropometric_prior_v22.py`, `test_kinematic_prior_zero_mean.py` |
| Neural bundle adjustment | `neural_bundle_adjustment_v21` | `test_neural_bundle_adjustment_v21.py`, `test_neural_bundle_adjustment_identity.py` |
| Skeleton graph refiner | `skeleton_graph_residual_refiner`, `kinematic_chain_graph_refiner` | `test_skeleton_graph_residual_refiner.py`, `test_kinematic_chain_graph_refiner.py` |
| Variable-view inference | `variable_view_inference`, `variable_view_set_aggregator` | `test_variable_view_inference_hardened.py` |
| Visibility / occlusion | `visibility_gated_fusion_v2`, `ray_attention_temporal_crossview_residual_principal_point_visibility_transformer_model` | `test_visibility_gated_fusion_v2.py`, `test_occlusion_robust_visibility_transformer.py` |

## Test gaps by category

### Camera / geometry helpers

- `motionflow_mv/fusion/camera_centric_coordinate_transform.py`
- `motionflow_mv/fusion/camera_conditioned_view_embedding.py`
- `motionflow_mv/fusion/camera_positional_encoding.py`
- `motionflow_mv/fusion/intrinsic_correction.py`
- `motionflow_mv/fusion/differentiable_bundle_adjustment.py`
- `motionflow_mv/fusion/dynamic_view_selection_gate.py`

### Robust triangulation

- `motionflow_mv/fusion/robust_triangulation.py`
- `motionflow_mv/fusion/robust_triangulation_baseline.py`
- `motionflow_mv/fusion/robust_triangulation_baseline_module.py`
- `motionflow_mv/fusion/robust_triangulation_module.py`

### Principal-point ray-attention variants

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_attention_entropy_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_visibility_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bundle_adjustment_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_camera_centric_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_camera_centric_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_canonical_skeleton_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_completion_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_dynamic_gate_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_skeleton_residual_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_hard_negative_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_multiperson_assoc_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_physics_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_refined_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_splat_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_visibility_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_visibility_transformer_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_crossview_contrast_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_module.py`

### Ray-attention v2/v3/v4 and temporal variants

- `motionflow_mv/fusion/ray_attention_v2_model.py`
- `motionflow_mv/fusion/ray_attention_v3_model.py`
- `motionflow_mv/fusion/ray_attention_v4_model.py`
- `motionflow_mv/fusion/ray_attention_crossview_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_model_v3.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_v4_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_v4_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_residual_learned_tri_v1_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_mixed_residual_principal_point_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_mixed_residual_v1.py`
- `motionflow_mv/fusion/ray_attention_temporal_model_mixed_v1.py`
- `motionflow_mv/fusion/ray_attention_temporal_learned_tri_v1.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_factorized_residual_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_adaptive_view_selection_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_campe_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`
- `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model.py`

### Cross-view attention

- `motionflow_mv/fusion/cross_view_spatial_pyramid.py`
- `motionflow_mv/fusion/cross_view_visibility_transformer.py`
- `motionflow_mv/fusion/epipolar_attention_bias.py`
- `motionflow_mv/fusion/epipolar_transformer_bias.py`
- `motionflow_mv/fusion/graph_joint_attention_v2.py`

### Residual / temporal refiners

- `motionflow_mv/fusion/residual_refiner_module.py`
- `motionflow_mv/fusion/temporal_refiner.py`
- `motionflow_mv/fusion/temporal_refiner_module.py`
- `motionflow_mv/fusion/canonical_skeleton_residual_refiner.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_module.py`

### Multiperson / association

- `motionflow_mv/fusion/multiperson_association_graph.py`
- `motionflow_mv/fusion/multi_task_shape_pose.py`

### Other high-level fusion models

- `motionflow_mv/fusion/action_aware_principal_point_model.py`
- `motionflow_mv/fusion/adaptive_hierarchical_multiscale_fusion.py`
- `motionflow_mv/fusion/semantic_action_conditional_fusion_model.py`
- `motionflow_mv/fusion/domain_adaptation_wrapper.py`
- `motionflow_mv/fusion/perceiver_view_aggregator.py`
- `motionflow_mv/fusion/variable_view_set_aggregator.py`
- `motionflow_mv/fusion/attention_fusion_v2_module.py`
- `motionflow_mv/fusion/attention_model_v2.py`
- `motionflow_mv/fusion/prototypes/trainer_optim_utils.py`

## Critical gaps for current experiments

1. **Bundle-adjustment variants.** `differentiable_bundle_adjustment.py` and the robust bundle-adjustment variants have no direct unit tests. Only `neural_bundle_adjustment_v21.py` is well tested.
2. **Robust triangulation.** `robust_triangulation*.py` are relied on by the extended robustness matrix and omniview v5 robust DLT, but none have targeted tests for reweighting logic or fallback behaviour.
3. **Principal-point ray-attention variants.** Many v23/v24 candidate models (epipolar, graph skeleton, dynamic gate, bayesian tri, spatial pyramid, camera-centric, etc.) only receive generic attention tests or are exercised indirectly by `test_iter14_models_train_step.py`.
4. **Camera / geometry helpers.** `camera_positional_encoding.py`, `camera_conditioned_view_embedding.py`, `camera_centric_coordinate_transform.py`, and `intrinsic_correction.py` are untested.
5. **Multiperson and association.** `multiperson_association_graph.py` and `multi_task_shape_pose.py` have no direct tests.
6. **Residual / temporal refiners.** `temporal_refiner*.py`, `residual_refiner_module.py`, `canonical_skeleton_residual_refiner.py`, and `semantic_action_conditional_fusion_model.py` lack targeted tests.
7. **Domain / augmentation wrappers.** `domain_adaptation_wrapper.py`, `synchronized_multiview_2d_aug.py` (prototype) only have integration-level tests.
8. **Evaluation smoke tests.** `test_eval_omniview_fusion_v*.py` check that scripts run, but do not assert metric regression thresholds.

## Recommendations

- Add parametrized unit tests for the `robust_triangulation*` family that verify reweighting against synthetic outliers and variable-view masks.
- Test `differentiable_bundle_adjustment.py` identity / first-derivative properties in isolation.
- Create a single model-variant test matrix for principal-point ray-attention models (forward/backward with variable views and camera perturbations) rather than one test per variant.
- Add tests for camera positional encoding and camera-conditioned embeddings to catch shape/gradient regressions in v23/v24 camera-aware fusion.
- Add a regression test for eval outputs that asserts an upper bound on MPJPE for a fixed smoke dataset.
- Harden variable-view and occlusion paths with assertions of valid output shape and finite-ness for the untested `variable_view_set_aggregator` and `multiperson_association_graph`.

## Modules with no direct test (full list)

- `motionflow_mv/fusion/action_aware_principal_point_model.py`
- `motionflow_mv/fusion/adaptive_hierarchical_multiscale_fusion.py`
- `motionflow_mv/fusion/attention_fusion_v2_module.py`
- `motionflow_mv/fusion/attention_model_v2.py`
- `motionflow_mv/fusion/camera_centric_coordinate_transform.py`
- `motionflow_mv/fusion/camera_conditioned_view_embedding.py`
- `motionflow_mv/fusion/camera_positional_encoding.py`
- `motionflow_mv/fusion/canonical_skeleton_residual_refiner.py`
- `motionflow_mv/fusion/cross_view_spatial_pyramid.py`
- `motionflow_mv/fusion/cross_view_visibility_transformer.py`
- `motionflow_mv/fusion/differentiable_bundle_adjustment.py`
- `motionflow_mv/fusion/domain_adaptation_wrapper.py`
- `motionflow_mv/fusion/dynamic_view_selection_gate.py`
- `motionflow_mv/fusion/epipolar_attention_bias.py`
- `motionflow_mv/fusion/epipolar_transformer_bias.py`
- `motionflow_mv/fusion/graph_joint_attention_v2.py`
- `motionflow_mv/fusion/intrinsic_correction.py`
- `motionflow_mv/fusion/multi_task_shape_pose.py`
- `motionflow_mv/fusion/multiperson_association_graph.py`
- `motionflow_mv/fusion/perceiver_view_aggregator.py`
- `motionflow_mv/fusion/prototypes/trainer_optim_utils.py`
- `motionflow_mv/fusion/ray_attention_crossview_model.py`
- `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_factorized_residual_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_adaptive_view_selection_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_campe_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_attention_entropy_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_visibility_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bundle_adjustment_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_camera_centric_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_camera_centric_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_canonical_skeleton_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_completion_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_dynamic_gate_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_hard_negative_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_multiperson_assoc_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_refined_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_splat_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_learned_tri_v1.py`
- `motionflow_mv/fusion/ray_attention_temporal_mixed_residual_principal_point_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_mixed_residual_v1.py`
- `motionflow_mv/fusion/ray_attention_temporal_model_mixed_v1.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_adaptive_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_adaptive_softgate_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_adaptive_softgate_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_graph_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_model_v3.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_module.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_v4_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_residual_learned_tri_v1_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_v4_model.py`
- `motionflow_mv/fusion/ray_attention_v2_model.py`
- `motionflow_mv/fusion/ray_attention_v3_model.py`
- `motionflow_mv/fusion/ray_attention_v4_model.py`
- `motionflow_mv/fusion/residual_refiner_module.py`
- `motionflow_mv/fusion/robust_triangulation.py`
- `motionflow_mv/fusion/robust_triangulation_baseline.py`
- `motionflow_mv/fusion/robust_triangulation_baseline_module.py`
- `motionflow_mv/fusion/robust_triangulation_module.py`
- `motionflow_mv/fusion/semantic_action_conditional_fusion_model.py`
- `motionflow_mv/fusion/temporal_refiner.py`
- `motionflow_mv/fusion/temporal_refiner_module.py`
- `motionflow_mv/fusion/variable_view_set_aggregator.py`

---

_This gap analysis is intended to guide the next round of test additions before the ICRA/CVPR 2027 submission._
