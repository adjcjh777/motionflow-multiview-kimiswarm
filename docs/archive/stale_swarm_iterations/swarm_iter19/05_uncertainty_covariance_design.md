# Subtask 05 — Uncertainty / Covariance Design

## Summary

This subtask reviews how the project represents and exploits per-view, per-joint image-space uncertainty in multi-view triangulation. A well-designed covariance/precision head should (1) downweight noisy or occluded views, (2) provide a calibrated confidence signal, and (3) feed a geometrically correct weighted DLT. The current flagship `OmniMultiViewFusionV2` already contains a Cholesky-factor covariance head inherited from the Bayesian Tri v2 lineage; the open question is whether it is being used to its full potential, especially now that the `graph_num_layers=0` ablation is running.

## Current state

* **Standalone module** `motionflow_mv/fusion/uncertainty_weighted_triangulation.py:21-307` implements a differentiable, anisotropic covariance-weighted DLT. It supports both `covariances` and `precisions` as 2×2 matrices and a learnable `UncertaintyWeightedTriangulation` module that predicts diagonal covariances.
* **Isotropic residual uncertainty** `motionflow_mv/models/crossview_residual_uncertainty.py:68-185` predicts a per-view log-variance, converts it to a scalar precision weight, and adds an auxiliary reprojection NLL loss.
* **Bayesian Tri v2 model** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py:117-499` extends the cross-view residual model with:
  * A Cholesky-factor head (`covariance_head`, line 182) that predicts `[l_xx, l_xy, l_yy]` per `(view, joint)` and builds a 2×2 lower-triangular `L`.
  * A scalar precision weight derived from `1 / (l_xx * l_yy)` used during DLT (line 300-302).
  * Adaptive Gauss-Newton refinement with a learned per-joint damping (line 318-334).
  * An epipolar-consistency auxiliary loss weighted by the harmonic mean of covariance determinants (line 215-239).
* **OmniMultiViewFusionV2** `motionflow_mv/fusion/omniview_fusion_v2.py:53-412` folds the Bayesian Tri v2 covariance head, visibility gating, graph-joint attention, and spatiotemporal attention into one model. The no-graph ablation is currently training with `graph_num_layers=0` (`scripts/run_omniview_fusion_v2_full_wsl.sh:28`), targeting `outputs/omniview_fusion_v2_d128_no_graph.pth`.
* **Training supervision** `experiments/train_omniview_fusion_v2_mpiinf3dhp.py:301-343` uses the predicted `L` in a reprojection NLL (`uncertainty_nll_loss`) and adds a scalar precision factor to the triangulation weight, but the DLT itself does not apply the full precision matrix.

## Key findings

1. **Scalar precision is a simplification.** In both `omniview_fusion_v2.py:299-301` and `ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py:299-302`, the anisotropic 2×2 covariance is reduced to `precision = 1/(l_xx * l_yy)`. The off-diagonal `l_xy` and the orientation of the covariance ellipse are ignored during triangulation, so the DLT solver is not Mahalanobis-optimal.
2. **The standalone module already supports full precision-DLT.** `uncertainty_weighted_triangulation.py:46-81` Cholesky-factors a 2×2 covariance and uses `W = S^{-1}` so that `W^T W = Σ^{-1}`; this is exactly the machinery needed to embed the full precision matrix into the DLT rows. It is not currently used by OmniMultiViewFusionV2.
3. **Covariance is only partially supervised.** The NLL loss (`uncertainty_nll_loss`) is sound, but there is no calibration check (e.g., predicted 95 % coverage vs empirical), no temporal smoothness prior on `L`, and no cross-view consistency term beyond the epipolar loss.
4. **The epipolar consistency loss is heuristic.** It weights pairwise epipolar distance by `1/(det_src + det_dst)` (`ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py:230-238`), but `compute_epipolar_distance` is not itself uncertainty-aware in the sense of a Sampson error with covariance propagation.
5. **No-graph ablation may isolate the value of covariance vs. graph attention.** The running experiment trains with `graph_num_layers=0`. If it reaches near the full-graph version, covariance/visibility design is the dominant signal and graph attention can be treated as optional. If it lags, the graph component is load-bearing.
6. **Existing tests cover the basics.** `tests/test_uncertainty_weighted_triangulation.py` verifies that large covariances downweight noisy views and that gradients flow, but there is no test for full-matrix precision in the DLT or for calibration.

## Recommendations

1. **Use the full 2×2 precision inside the DLT.** Replace the scalar `precision` weight in `omniview_fusion_v2.py:299-310` and `ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py:299-312` with the Cholesky-based weighting already implemented in `uncertainty_weighted_triangulation.py:46-81`. This makes the triangulation step statistically consistent with the predicted covariances.
2. **Add an isotropic-vs-anisotropic ablation.** Run a short d=48, 10-epoch smoke with the covariance head constrained to diagonal-only (`l_xy` forced to 0) and compare clean MPJPE, robustness, and NLL against the full Cholesky head.
3. **Introduce a calibration diagnostic.** After each epoch, compute the empirical coverage of the predicted covariance ellipses on the validation set (e.g., percentage of reprojection residuals inside the 1σ and 2σ ellipses). Add it to the training log as a sanity metric.
4. **Smooth `L` temporally.** Add a lightweight temporal regularizer on consecutive frames' Cholesky parameters, or equivalently predict `L` from a temporal context rather than a single frame, to reduce jitter in the uncertainty estimates.
5. **Scope the no-graph ablation carefully.** Keep the current `graph_num_layers=0` run as-is, but add an explicit smoke/eval script that compares (a) full graph, (b) no graph, and (c) no graph + full precision-DLT, all with the same warm-start. This directly answers whether graph attention is load-bearing or merely a capacity multiplier.
6. **Fix epipolar loss geometry.** Replace the determinant-weighted ad-hoc term with a proper covariance-weighted Sampson or symmetric epipolar distance, or drop it if the full precision-DLT already enforces geometry.

## Open questions

* Does switching from scalar `1/det(Σ)` to full precision-DLT improve clean MPJPE on MPI-INF-3DHP, or does it mainly improve calibration/robustness?
* Is the predicted covariance well-calibrated? Do high-uncertainty predictions actually correspond to high reprojection error?
* How much of OmniMultiViewFusionV2's accuracy is due to the anisotropic covariance head versus the visibility head versus graph-joint attention? A one-at-a-time ablation is needed.
* Does the no-graph ablation (`outputs/omniview_fusion_v2_d128_no_graph.pth`) reach the 8.0 mm target? If so, should graph attention be deprioritized in favor of a leaner covariance-focused architecture?
* Would temporal smoothing of `L` hurt or help view-dropout robustness (`view_dropout_30` target ≤ 13 mm)?
