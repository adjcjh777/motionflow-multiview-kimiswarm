# v54 Design Space Synthesis

Synthesized from the 20 design proposals in `docs/swarm_iter28/proposals/` and their risk reports in `docs/swarm_iter28/reports/`. MotionFlow-MultiView is at the v53 Physical-Space Calibration milestone: v53 calibrates the v52 uncertainty-weighted triangulation output against floor-plane and canonical bone-length invariants. v54 should build on that warm-started backbone and tighten the *physical-space alignment* stage before any further domain/temporal expansion.

## 1. Executive Summary

The v53 PSC smoke is queued behind the v52 medium smoke. Once v53 is proven, the next bottleneck is that v53 applies a **single global residual MLP** and scalar gate; local errors (foot-floor penetration, over-stretched limbs, wrist/ankle jitter) are not corrected on a per-joint, per-body-part basis. v54 should therefore add a **Physical-Space Calibration v2 (PSC-v2)** module: a skeleton-graph, joint-level physical refiner that sits on top of v53, consumes v52 UWT weights as a robustness signal, and enforces floor/contact, bone-length, and temporal-continuity constraints while remaining identity-at-init.

The 20 proposals fall into five natural clusters:

1. **Physical / calibration refinements** (PSC-v2, bone-length-aware fusion, SMPL bridge, camera noise correction, differentiable bundle adjustment, implicit neural geometry, test-time self-refinement).
2. **Robust fusion / view weighting** (outlier-robust reliability, learned view selection, multi-scale geometry fusion, geometry-aware attention pooling, video feature extractor).
3. **Domain / dataset bridging** (adaptive low-rank per-domain fusion, cross-dataset domain bridge, domain-conditional normalization).
4. **Temporal / consistency** (multi-view temporal sync, temporal consistency loss, probabilistic pose forecasting).
5. **Self-supervised / model-based priors** (self-supervised multi-view pretraining, SMPL human-model bridge, reliability-guided pose mixup).

All 20 are identity-at-init by construction. The ranking below orders them by expected sparse/full-view MPJPE impact per unit risk, with strong preference for paper-aligned, low-risk candidates that extend the physical-space calibration story.

## 2. Ranking Table of the 20 v54 Proposals

| Rank | Proposal (file) | Key idea | Expected MPJPE impact | Risk | Warm-start / identity-at-init |
|---:|---|---|---|---|---|
| 1 | **Physical-Space Calibration v2** (`v54_physical_space_calibration_v2_v54.md`) | Skeleton-graph physical refiner on top of v53: floor/contact, per-domain bone scales, per-joint GNN residual. | Identity `< 0.1 mm`; smoke `−0.5 to −1.5 mm`; full `−0.8 to −2.0 mm`, larger on `@2/3`. | Medium | Yes — zero-init final layers, gate logit `−6.0`, bone log-scales init `0`. |
| 2 | **Outlier-Robust Reliability** (`v54_outlier_robust_reliability_v54.md`) | Learned Cauchy M-estimator refines v52 UWT weights before triangulation. | Smoke `−0.5 to −1.5 mm`; full `−0.8 to −2.0 mm`; sparse `−2 to −4 mm`. | Medium | Yes — residual gate `−6.0`, `γ ≈ 1` at init, final layers zero-init. |
| 3 | **Temporal Consistency Loss** (`v54_temporal_consistency_loss_v54.md`) | Loss-only velocity/acceleration regulariser weighted by v52/v53 uncertainty. | Smoke ≤ `0.5 mm`; full `−0.3 to −0.8 mm`; fast-motion `−1.0 to −1.5 mm`. | Low-Medium | Yes — loss-only, no forward change; weight ramped from zero. |
| 4 | **Bone-Length-Aware Fusion** (`v54_bone_length_aware_fusion_v54.md`) | Per-bone canonical length offset + gated residual on v53 pose. | Full `−0.3 to −0.8 mm`; sparse `−1.5 to −3.0 mm`. | Medium-High | Yes — canonical offset MLP zero-init, gate `−6.0`. |
| 5 | **Multi-View Temporal Sync** (`v54_multi_view_temporal_sync.md`) | Per-joint temporal attention over `(time, view)` tokens after v53. | Full `−0.5 to −1.0 mm`; sparse `−1.5 to −3.0 mm`. | Medium | Yes — zero-init residual layer, gate `−6.0`. |
| 6 | **Cross-Dataset Domain Bridge** (`v54_cross_dataset_domain_bridge_v54.md`) | Domain-conditional affine normalization into canonical pose space + refiner. | In-domain `±0.2 mm`; cross-domain `−1.0 to −2.5 mm`; sparse `−0.8 to −2.0 mm`. | Medium-High | Yes — zero-init affine/refiner, gate `−6.0`. |
| 7 | **Multi-Scale Geometry Fusion** (`v54_multi_scale_geometry_fusion_v54.md`) | Fuse joint/limb/body/scene tokens after v53 with cross-view attention. | Full `−0.5 to −1.0 mm`; `@2` `−2 to −4 mm`; `@3` `−1 to −2 mm`. | Medium-High | Yes — zero-init output layers, cross-scale gates `0`. |
| 8 | **Learned View Selection Policy** (`v54_learned_view_selection_policy.md`) | Differentiable top-K policy selecting view subsets on v52 weights. | Full `−1.0 to −2.0 mm`; sparse `−2 to −5 mm`. | Medium-High | Yes — zero logits + learned rescale, sigmoid warm-start. |
| 9 | **Test-Time Self-Refinement** (`v54_test_time_self_refinement_v54.md`) | Iterative learned refinement of v53 pose using reprojection + physical feedback. | Medium `−1 to −2 mm`; full `−2 to −4 mm`; 3DPW `−3 to −5 mm`. | Medium | Yes — correction/gate heads zero-init. |
| 10 | **Probabilistic Pose Forecasting** (`v54_probabilistic_pose_forecasting.md`) | Causal Gaussian forecast head + gated correction after v53. | Smoke ≤ `0.5 mm`; medium `−1 to −2 mm`; full `−2 to −4 mm`. | Medium | Yes — zero-init heads, gate `−6.0`. |
| 11 | **Reliability-Guided Pose Mixup** (`v54_reliability_guided_pose_mixup_v54.md`) | Mix v53 pose with learned canonical anchor conditioned on v52 reliability. | Full `−0.2 to −0.7 mm`; sparse `−1.0 to −2.5 mm`. | Medium | Yes — `α` clamped near `0` at init, residual gate `−6.0`. |
| 12 | **Self-Supervised Multi-View Pretraining** (`v54_self_supervised_multiview_pretraining_v54.md`) | Masked-view triangulation, cross-view feature consistency, temporal continuity. | Full `−0.5 to −1.5 mm`; sparse `3–6%` rel. | Medium | Yes — residual gate `−6.0`, loss warmup. |
| 13 | **Domain-Conditional Normalization** (`v54_domain_conditional_normalization.md`) | Per-domain affine on calibrated pose and v52 weights after v53. | Mixed-domain `−0.5 to −1.2 mm`; sparse improves. | Medium | Yes — zero-init MLP final layers, tanh-bounded affine. |
| 14 | **Differentiable Bundle Adjustment** (`v54_differentiable_bundle_adjustment.md`) | Jointly refine pose and camera parameters after v53. | Full `−0.4 to −1.0 mm`; sparse `−1.0 to −2.5 mm`. | Medium-High | Yes — zero-init correction MLPs, gated pose residual. |
| 15 | **Camera Noise Correction** (`v54_camera_noise_correction.md`) | Learned 2-D keypoint + camera parameter correction before v52. | Clean `0 to −0.3 mm`; noisy `−0.5 to −1.5 mm`; up to `−2.5 mm`. | Medium-High | Yes — bounded corrections, zero-init final layers, gate `−6.0`. |
| 16 | **Adaptive Low-Rank Per-Domain Fusion** (`v54_adaptive_lr_per_domain_v54.md`) | Low-rank per-domain adapters after ST transformer, gated by v52/v53 signals. | Full `−0.3 to −0.8 mm`; sparse `−1.0 to −2.5 mm`. | Medium | Yes — `B_d` zero-init, gate `−6.0`. |
| 17 | **Geometry-Aware Attention Pooling** (`v54_geometry_aware_attention_pooling.md`) | Geometry-biased cross-view attention before v52 UWT. | Full `−0.4 to −1.2 mm`; sparse larger. | Medium-High | Yes — zero-init output projection, `γ = 0` at init. |
| 18 | **Implicit Neural Geometry** (`v54_implicit_neural_geometry_v54.md`) | Spatio-temporal implicit neural geometry refinement after v53. | Full `−0.5 to −1.2 mm`; sparse `−1.0 to −2.5 mm`; 3DPW `−2 to −4 mm`. | Medium-High | Yes — final ING layers zero-init, gate `−6.0`. |
| 19 | **SMPL Human Model Bridge** (`v54_smpl_human_model_bridge_v54.md`) | SMPL body-model parameter prediction + gated residual after v53. | Smoke `−1 to −3 mm`; full `1–2%` gain. | Medium-High | Yes — final MLP zero-init, gate `−6.0`. |
| 20 | **Video Feature Extractor** (`v54_video_feature_extractor.md`) | Spatiotemporal feature refinement before v52. | Full `−0.8 to −2.0 mm`; sparse `−2 to −3 mm`. | Medium-High | Yes — zero-init output projections, global gate `0`. |

## 3. Top-1 Recommendation: Physical-Space Calibration v2 (PSC-v2)

**Module:** `PhysicalSpaceCalibrationV2V54` → `motionflow_mv/fusion/physical_space_calibration_v2_v54.py`

**Why:** v53 made the triangulated pose physically aware at a global level; the remaining gap is **local physical-space calibration**. PSC-v2 directly extends the paper narrative (multi-view fusion and calibration → physical-space alignment → optimized motionflow) by adding a skeleton-graph, joint-level refiner on top of v53. It is the lowest-risk, most paper-aligned candidate: it reuses v52 UWT weights as robustness signals, keeps all v53 machinery intact, and is fully identity-at-init. Because it refines the pose *after* v53, it also leaves the v53 checkpoint path untouched until the new heads learn.

PSC-v2 is the safest high-impact candidate among the v54 proposals. It introduces a single shallow GNN over the kinematic chain, learnable per-domain canonical bone log-scales, and a small set of physical losses — all of which can be unit-tested for identity-at-init and gradient stability. The risk register highlights GNN over-smoothing, floor/contact assumptions, canonical bone conflicts, and warm-start failure, but each has a concrete mitigation (shallow graph, velocity-gated contact, per-domain scales, zero-initialized final layers). The module can be smoke-tested on the local RTX 4090 in a few hours and, if it passes, dropped into the A800 queue with minimal trainer/data-loader changes.

## 4. Justification

The v53 synthesis identified physical-space calibration as the highest-priority target. v53 addressed it with a global floor/bone calibration and a gated residual MLP. The next logical step is to make that calibration **local and anatomically structured**: feet should respect the floor only when moving slowly, forearm/humerus bones should maintain plausible lengths, and corrections should propagate along the skeleton graph rather than being broadcast uniformly. PSC-v2 does exactly that: a floor/contact head estimates a per-frame floor height from UWT-weighted foot joints, a bone-length head learns per-domain canonical log-scales, and a skeleton-graph residual refiner fuses physical hints (floor distance, bone-length residual, reprojection error, temporal velocity) into bounded per-joint corrections. Because the GNN output layer, bone-scale output layer, and residual gate are all zero-initialized, a v53 checkpoint loads unchanged and the baseline is preserved until the new heads learn.

PSC-v2 is the best launchpad for the other v54 candidates. Once the pose is locally physically calibrated, the outlier-robust reliability, bone-length-aware fusion, and temporal-consistency proposals have a cleaner signal to work from. It also naturally precedes any SMPL/model-based bridge (avoids over-constraining before local calibration is proven) and any domain-bridging module (a better-calibrated pose gives the canonical pose space a more stable target). Choosing PSC-v2 first therefore preserves optionality while directly advancing the paper narrative.

## 5. Runner-Up Candidates for v55/v56

1. **Outlier-Robust Reliability (OR2)** — A learned Cauchy M-estimator that refines v52 UWT weights before triangulation. Strong synergy with PSC-v2: removing gross outlier views before physical calibration makes both tasks easier. Run it once PSC-v2 has stabilized the local pose.

2. **Bone-Length-Aware Fusion (BLAF)** — A per-bone canonical-length offset + gated residual that complements PSC-v2's bone-scale head with a more explicit bone-length prior. Medium-high risk because it can double-count v53 bone constraints; ablate bone loss weights carefully.

3. **Temporal Consistency Loss (TCL)** — A loss-only temporal smoothness term weighted by v52/v53 uncertainty. Lowest implementation cost and strong synergy with PSC-v2; no new forward module, so it can be stacked quickly as a v55 ablation.

4. **Multi-View Temporal Sync (MVTS)** — Per-joint temporal attention over `(time, view)` tokens after v53. Good for borrowing clean views across time, but adds memory; run after the local physical calibration is proven.

5. **Cross-Dataset Domain Bridge (CDDB)** — Domain-conditional affine normalization + refiner after v53. Best suited to v56 once the in-domain physical pose is strong and the remaining gap is cross-dataset bias removal.
