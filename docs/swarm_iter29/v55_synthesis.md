# v55 Design Space Synthesis

Synthesized from the 20 design proposals in `docs/swarm_iter29/proposals/`. MotionFlow-MultiView is at the v54 Physical-Space Calibration v2 milestone: v54 made the triangulated pose physically consistent at a joint/skeleton-graph level. v55 should now tighten the *input to that calibrated pipeline*—the per-view reliability weights that feed v52 Uncertainty-Weighted Triangulation—while keeping the proven v53/v54 physical calibration stack intact.

## 1. Executive Summary

The v54 PSC-v2 smoke is queued behind the v53 PSC medium smoke. Once v54 is proven, the next bottleneck is that the quality of the physically-calibrated pose is still bounded by the upstream per-view weights produced by v45 geometry fusion. A single outlier view can dominate triangulation, and this is most visible in sparse-view regimes (`MPJPE@2/3`). v55 should therefore add an **Outlier-Robust Reliability (OR2)** module: a lightweight, identity-at-init Cauchy M-estimator that refines v45 weights before they reach v52 UWT. It preserves the v54 checkpoint at init, is self-contained before the physical calibration stages, and has the largest sparse-view upside of the v55 candidate set.

The 20 proposals fall into five natural clusters:

1. **Physical / calibration refinements** — Bone-Length-Aware Fusion, Physical-Contact Dynamics, SMPL Human Model Bridge, Test-Time Self-Refinement, Differentiable Bundle Adjustment, Camera Noise Correction, Implicit Neural Geometry, Skeleton-Graph Uncertainty Gating.
2. **Robust fusion / view weighting** — Outlier-Robust Reliability, Learned View Selection Policy, Geometry-Aware Attention Pooling, Video Feature Extractor, Skeleton-Graph Uncertainty Gating (also has a graph component).
3. **Temporal / consistency** — Multi-View Temporal Sync, Probabilistic Pose Forecasting, Temporal Consistency Loss.
4. **Domain / dataset bridging** — Domain-Conditional Normalization v2, Cross-Dataset Domain Bridge, Adaptive Low-Rank Per-Domain Fusion.
5. **Self-supervised / model-based priors** — Self-Supervised Multi-View Pretraining, SMPL Human Model Bridge, Reliability-Guided Pose Mixup, Probabilistic Pose Forecasting.

All 20 are identity-at-init by construction. The ranking below orders them by expected sparse/full-view MPJPE impact per unit implementation risk, with strong preference for paper-aligned, low-risk candidates that do not duplicate the v54 physical calibration story.

## 2. Ranking Table of the 20 v55 Proposals

| Rank | Proposal (file) | Key idea | Expected MPJPE impact | Risk | Warm-start / identity-at-init |
|---:|---|---|---|---|---|
| 1 | **Outlier-Robust Reliability (OR2)** (`v55_outlier_robust_reliability.md`) | Cauchy M-estimator refines v45 weights before v52 UWT, using geometry/feature bias. | Full `−0.8 to −2.0 mm`; sparse `−2 to −4 mm`; 3DPW `−1 to −3 mm`. | Medium | Yes — residual gate `−6.0`, weights clamped, zero-init final layer. |
| 2 | **Temporal Consistency Loss (TCL)** (`v55_temporal_consistency_loss.md`) | Loss-only velocity/acceleration regulariser weighted by v52/v54 uncertainty. | Full `−0.3 to −0.8 mm`; fast motion `−1.0 to −1.5 mm`; sparse `−0.2 to −0.6 mm`. | Low | Yes — no forward parameters; weight ramped from zero. |
| 3 | **Multi-View Temporal Sync (MVTS)** (`v55_multi_view_temporal_sync.md`) | Per-joint `(time, view)` attention after v54 to borrow clean views across time. | Full `−0.5 to −1.0 mm`; sparse `−1.5 to −3.0 mm`. | Medium | Yes — zero-init correction MLP and gate `−6.0`. |
| 4 | **Geometry-Aware Attention Pooling (GAAP)** (`v55_geometry_aware_attention_pooling.md`) | Geometry-biased cross-view attention before v52 UWT. | Full `−0.4 to −1.2 mm`; sparse `−1.0 to −2.5 mm`. | Medium-High | Yes — zero-init output projection, gate `−6.0`, geometry scalar `0.0`. |
| 5 | **Test-Time Self-Refinement (TTSR)** (`v55_test_time_self_refinement.md`) | Iterative learned corrector using reprojection + physical feedback after v54. | Full `−0.8 to −2.0 mm`; sparse `−1.5 to −3.5 mm`; 3DPW `−2.0 to −4.0 mm`. | Medium-High | Yes — zero-init correction layer, gate `−6.0`, fixed 3-step loop. |
| 6 | **Learned View Selection Policy (LVSP)** (`v55_learned_view_selection_policy.md`) | Differentiable view-subset policy reweighting before a gated re-triangulation. | Full `−0.8 to −1.5 mm`; `@2` `−2 to −5 mm`. | Medium-High | Yes — zero-init final MLP layer, high-temperature anneal, gate `−6.0`. |
| 7 | **Video Feature Extractor (VFE)** (`v55_video_feature_extractor.md`) | Spatiotemporal refinement of raw 2D keypoints/confidences before v25/v45. | Full `−0.8 to −2.0 mm`; sparse `−2.0 to −4.0 mm`. | Medium-High | Yes — zero-init output projections, gate `−6.0`. |
| 8 | **Skeleton-Graph Uncertainty Gating (SGUG)** (`v55_skeleton_graph_uncertainty_gating.md`) | Kinematic graph refiner gated by per-joint v52 uncertainty after v54. | Full `−0.5 to −1.5 mm`; sparse `−1.0 to −3.0 mm`; 3DPW `−1.0 to −2.5 mm`. | Medium-High | Yes — zero-init correction layer, gate `−6.0`. |
| 9 | **Self-Supervised Multi-View Pretraining (SSMP)** (`v55_self_supervised_multiview_pretraining.md`) | Masked-view triangulation + cross-view feature consistency + temporal smoothness. | Full `−0.5 to −1.5 mm`; sparse `3–6%` rel. | Medium | Yes — zero-init final layers, gate `−6.0`, loss warmup. |
| 10 | **Probabilistic Pose Forecasting (PPF)** (`v55_probabilistic_pose_forecasting.md`) | Causal Gaussian forecast head + gated correction after v54. | Full `−1.0 to −2.5 mm`; sparse `−1.0 to −3.0 mm`. | Medium | Yes — zero-init mean MLP, gate `−6.0`. |
| 11 | **Implicit Neural Geometry (ING)** (`v55_implicit_neural_geometry.md`) | Per-joint implicit neural field refinement after v54 using ray/UWT features. | Full `−0.4 to −1.0 mm`; sparse `−1.0 to −2.5 mm`; 3DPW `−1.5 to −3.5 mm`. | Medium-High | Yes — zero-init final layers, gate `−6.0`. |
| 12 | **Reliability-Guided Pose Mixup (RGPM)** (`v55_reliability_guided_pose_mixup.md`) | Blend v54 pose with learned per-domain canonical anchor by reliability. | Full `−0.2 to −0.7 mm`; sparse `−1.0 to −2.5 mm`. | Medium | Yes — `α` gate `−6.0`, zero-init anchor head. |
| 13 | **Physical-Contact Dynamics (PCD)** (`v55_physical_contact_dynamics.md`) | Contact-state head + zero-velocity/no-penetration loss after v54. | Full `−0.5 to −1.2 mm`; sparse `−1.0 to −2.5 mm`. | Medium | Yes — zero-init contact/correction heads, gate `−6.0`. |
| 14 | **Bone-Length-Aware Fusion (BLAF)** (`v55_bone_length_aware_fusion.md`) | Per-bone canonical length offset + gated residual on v54 pose. | Full `−0.3 to −0.8 mm`; sparse `−1.5 to −3.0 mm`. | Medium-High | Yes — zero-init offset and gate. |
| 15 | **Camera Noise Correction (CNC)** (`v55_camera_noise_correction.md`) | Learned 2D keypoint + camera correction before v52 UWT. | Full `−0.2 to −0.8 mm`; sparse `−1.0 to −2.5 mm`; noisy up to `−2.5 mm`. | Medium-High | Yes — zero-init final layers, gate `−6.0`. |
| 16 | **Differentiable Bundle Adjustment (DBA)** (`v55_differentiable_bundle_adjustment.md`) | Joint pose + camera refinement after v54. | Full `−0.4 to −1.0 mm`; sparse `−1.0 to −2.5 mm`. | Medium-High | Yes — zero-init correction MLPs, gated pose residual. |
| 17 | **Domain-Conditional Normalization v2 (DCN)** (`v55_domain_conditional_normalization.md`) | Per-domain affine normalization of calibrated pose and v52 weights. | Mixed-domain `−0.5 to −1.2 mm`; sparse small. | Medium | Yes — zero-init final MLPs, bounded affine. |
| 18 | **Cross-Dataset Domain Bridge (CDDB)** (`v55_cross_dataset_domain_bridge.md`) | Dataset-agnostic canonical pose space + MMD alignment after v54. | Cross-domain `−1.0 to −2.5 mm`; full `−0.3 to −0.8 mm`. | Medium-High | Yes — zero-init affine/refiner, gate `−6.0`. |
| 19 | **Adaptive Low-Rank Per-Domain Fusion (ALRPD)** (`v55_adaptive_low_rank_per_domain.md`) | Low-rank per-domain adapters after v54, gated by v52 reliability. | Full `−0.3 to −0.8 mm`; sparse `−1.0 to −2.5 mm`. | Medium | Yes — `B_d` zero-init, gate `−6.0`. |
| 20 | **SMPL Human Model Bridge (SMB)** (`v55_smpl_human_model_bridge.md`) | SMPL parameter prediction + gated residual after v54. | Full `−0.5 to −1.5 mm`; sparse `−1.0 to −3.0 mm`. | Medium-High | Yes — zero-init regressor/residual layers, gate `−6.0`. |

## 3. Top-1 Recommendation: Outlier-Robust Reliability (OR2)

**Module:** `OutlierRobustReliabilityV55` → `motionflow_mv/fusion/outlier_robust_reliability_v55.py`

**Why:** v54 PSC-v2 made the pose physically consistent at the output stage; the remaining gap is at the *input stage*: the per-view weights from v45 geometry fusion still let a single bad view dominate triangulation. OR2 is the lowest-risk, highest-leverage place to intervene next because it sits immediately after v45 and before v52 UWT, cleaning the signal that every downstream physical/temporal/SEFH head consumes. It is purely additive, identity-at-init, and directly improves sparse-view robustness. It also naturally precedes any further physical refinement (v55 BLAF/PCD/TTSR) because those modules benefit from cleaner triangulation weights.

OR2 is the safest high-impact candidate among the v55 proposals. It introduces a single shallow per-token MLP and a Cauchy M-estimator, can be unit-tested for weight sanity and outlier rejection, and requires only the existing v45/v52 interface. The risk register highlights gate failure, Cauchy scale drift, and sparse-view degeneracy, but each has a concrete mitigation (gate logit `−6.0`, bounded `γ`, `min_weight`, and masking the top-`min_views`). The module can be smoke-tested on the local RTX 4090 in a few hours and, if it passes, dropped into the A800 queue with minimal trainer/data-loader changes.

## 4. Justification

The v54 synthesis identified local physical-space calibration as the highest-priority target; v54 extended v53 with a skeleton-graph, joint-level refiner. The next logical step is to ensure the *inputs* to that calibrated stack are robust to outliers. v45 geometry fusion produces per-view joint weights, and v52 UWT further refines them, but neither stage is explicitly designed to down-weight gross outlier views before triangulation. OR2 fills that gap: a per-(view, joint) outlier score under a Cauchy kernel down-weights views whose reprojection/ray/epipolar features disagree with the consensus, and a residual gate preserves the v54 baseline until the network learns.

Because OR2 refines *weights* rather than the pose directly, it does not duplicate v54's floor/bone/contact constraints; instead it gives v52/v53/v54 a cleaner pose to refine. It also leaves downstream physical modules untouched, so v55 BLAF/PCD/TTSR can be stacked on top of OR2 later with minimal interaction risk. Choosing OR2 first therefore preserves optionality while directly improving the reliability of the core multi-view fusion stage.

## 5. Runner-Up Candidates for v56/v57

1. **Temporal Consistency Loss (TCL)** — A loss-only velocity/acceleration regulariser weighted by v52/v54 uncertainty. It is the lowest-risk v55 proposal (no forward parameters) and should be stacked quickly as a v56 ablation once OR2 is proven.

2. **Multi-View Temporal Sync (MVTS)** — Per-joint `(time, view)` attention after v54. Strong synergy with OR2: cleaner weights make temporal borrowing more reliable. Run it after OR2 has stabilized the per-frame weight distribution.

3. **Geometry-Aware Attention Pooling (GAAP)** — Geometry-biased cross-view attention before v52 UWT. Complements OR2 by refining per-view feature tokens; risk is slightly higher because it changes the feature representation used by v52.

4. **Test-Time Self-Refinement (TTSR)** — Iterative learned refinement after v54. High potential on cross-domain/sparse data, but higher risk due to the unrolled loop; best attempted once OR2 + TCL are in place.

5. **Bone-Length-Aware Fusion (BLAF)** / **Physical-Contact Dynamics (PCD)** — Both extend the v54 physical-calibration story. Run them once the upstream weight quality (OR2) is proven, to avoid double-counting on noisy triangulation.
