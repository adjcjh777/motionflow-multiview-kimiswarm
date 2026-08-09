# v52 Design Space Synthesis

Synthesized from the 20 design proposals in `docs/swarm_iter26/proposals/` and their risk reports in `docs/swarm_iter26/reports/`. The project is at the v51 Cross-Domain Sparse-View Reliability milestone: v50 SEFH and v51 CDSVR are implemented, with A800 full runs queued. v52 should build on the proven v45/v46/v47/v48/v50/v51 backbone and attack the next bottleneck in the paper pipeline — the triangulation and calibration step that turns per-view 2-D evidence into a 3-D pose.

## 1. Executive Summary

v51 made per-view reliability **domain-aware**. The remaining gap is that the fusion/triangulation stage still treats geometry weights as either hand-designed or learned only implicitly. v52 should therefore make triangulation itself **uncertainty-aware** while keeping every candidate warm-startable/identity-at-init so it can be smoke-tested on the local RTX 4090 without waiting for A800 results.

The 20 proposals fall into five natural clusters:

1. **Triangulation / view weighting** (uncertainty-weighted triangulation, learned view selection, adaptive sparse-view dropout, outlier-robust reliability, geometry-aware attention pooling, multi-scale geometry fusion).
2. **Calibration / geometry** (camera noise correction, physical-space calibration, differentiable bundle adjustment).
3. **Domain / normalization** (cross-dataset domain bridge, domain-conditional normalization).
4. **Physical / parametric priors** (bone-length-aware fusion, SMPL human-model bridge, implicit neural geometry).
5. **Temporal / data / test-time** (multi-view temporal sync, temporal consistency loss, video feature extractor, self-supervised pretraining, reliability-guided pose mixup, test-time self-refinement).

All 20 are identity-at-init by construction; the ranking below orders them by expected sparse/full-view MPJPE impact per unit risk.

## 2. Ranking Table of the 20 v52 Proposals

Ranked by expected MPJPE impact per unit risk, with warm-start/identity-at-init feasibility noted.

| Rank | Proposal (file) | Key idea | Expected MPJPE impact | Risk | Warm-start / identity-at-init |
|---:|---|---|---|---|---|
| 1 | **Uncertainty-Weighted Triangulation** (`v52_uncertainty_weighted_triangulation_v52.md`) | Learn a per-view per-joint precision that drives a differentiable weighted DLT before downstream refinement. | `MPJPE@full` −2 to −5 mm; `MPJPE@2/3` larger; cross-domain −3 to −6 mm. | Low-Medium | Yes — zero-initialized precision MLP and residual. |
| 2 | **Outlier-Robust Reliability** (`v52_outlier_robust_reliability.md`) | Refine v51 reliability with Tukey/Huber M-estimator and physical cues (bone/floor). | `MPJPE@full` −1 to −4 mm; sparse-view larger. | Medium | Yes — final projections zero-initialized. |
| 3 | **Geometry-Aware Attention Pooling** (`v52_geometry_aware_attention_pooling.md`) | Geometry-biased cross-view attention after the ST transformer, using rays and epipolar distance. | `MPJPE@2/3` −3 to −6 mm; full-view −1 to −3 mm. | Medium | Yes — zero-initialized residual gate. |
| 4 | **Multi-Scale Geometry Fusion** (`v52_multi_scale_geometry_fusion_v52.md`) | Joint/part/body scale cross-view attention with learnable soft assignment. | `MPJPE@2/3` −1.5 to −3 mm; full-view −0.3 to −0.8 mm. | Medium | Yes — zero-initialized residual mixer. |
| 5 | **Learned View Selection Policy** (`v52_learned_view_selection_policy.md`) | Differentiable policy that selects camera subsets before triangulation. | `MPJPE@full` −1 to −3 mm; `MPJPE@2/3` closes gap by 5–10 %. | Medium | Yes — sigmoid mask ≈ 1 at init. |
| 6 | **Adaptive Sparse-View Dropout** (`v52_adaptive_sparse_view_dropout_v52.md`) | Learned per-(view,joint) dropout gate replacing v46 uniform dropout. | `MPJPE@2/3` 5–10 % lower than v46; full-view neutral. | Medium | Yes — budget initialized to all views. |
| 7 | **Camera Noise Correction** (`v52_camera_noise_correction.md`) | Learned correction head for intrinsics/extrinsics before triangulation. | Clean data neutral to −1 mm; synthetic noise −3 to −6 mm. | Medium | Yes — residual gate initialized near zero. |
| 8 | **Physical-Space Calibration** (`v52_physical_space_calibration.md`) | Joint camera/pose refinement using reprojection + bone + floor priors. | `MPJPE@full` −0.5 to −2 mm; noisy calibration larger. | Medium | Yes — zero-initialized camera/pose heads. |
| 9 | **Cross-Dataset Domain Bridge** (`v52_cross_dataset_domain_bridge_v52.md`) | Domain-conditional FiLM + cross-domain prototype attention for domain-invariant features. | Cross-dataset −3 to −6 mm; full-view ±0.5 mm. | Medium-High | Yes — geometry-preserving residual gate. |
| 10 | **Domain-Conditional Normalization** (`v52_domain_conditional_normalization.md`) | Domain- and view-count-conditional affine over feature tokens before the ST transformer. | Cross-domain −0.5 to −1.5 mm; sparse-view modest. | Low-Medium | Yes — zero-initialized output projections. |
| 11 | **Bone-Length-Aware Fusion** (`v52_bone_length_aware_fusion.md`) | Inject bone-length prior between triangulation and physical-space alignment. | `MPJPE@2/3` −1.5 to −3.5 mm; full-view −0.3 to −1.0 mm. | Medium | Yes — zero-initialized length/gate heads. |
| 12 | **Test-Time Self-Refinement** (`v52_test_time_self_refinement_v52.md`) | Learned post-hoc pose refiner using reprojection/bone/temporal cues. | WebBridge/H36M −2 to −4 mm; MPI/3DPW −3 to −6 mm. | Medium | Yes — zero-initialized correction/gate. |
| 13 | **Temporal Consistency Loss** (`v52_temporal_consistency_loss.md`) | Multi-scale Huber smoothness + bone-length consistency added to training loss. | `MPJPE@2/3` −1 to −2 mm; full-view −0.3 to −0.8 mm. | Low-Medium | Yes — loss weight ramped from zero. |
| 14 | **Multi-View Temporal Sync** (`v52_multi_view_temporal_sync.md`) | Learned per-view temporal warp + cross-view temporal attention before ST transformer. | Fast motion −2 to −5 mm; static neutral. | Medium | Yes — identity-init residual gate. |
| 15 | **Reliability-Guided Pose Mixup** (`v52_reliability_guided_pose_mixup_v52.md`) | Generate reliability-weighted pose candidates and fuse with per-joint transformer. | Sparse/cross-domain −2 to −5 mm; full-view −0.5 to −1.0 mm. | Medium | Yes — identity candidate + zero output. |
| 16 | **Video Feature Extractor** (`v52_video_feature_extractor_v52.md`) | Factorized temporal/cross-view/skeleton attention before triangulation. | Fast motion/noisy 2D −1 to −4 mm. | Medium-High | Yes — zero-initialized residual branch. |
| 17 | **Self-Supervised Multi-View Pretraining** (`v52_self_supervised_multiview_pretraining.md`) | Masked token reconstruction + contrastive view-consistency pretext task. | Sparse/cross-domain −2 to −4 mm; full-label modest. | High | Yes — identity residual projection. |
| 18 | **Differentiable Bundle Adjustment** (`v52_differentiable_bundle_adjustment.md`) | Learned init + unrolled Gauss-Newton/LM refinement of pose and cameras. | `MPJPE@2/3` −1 to −3 mm; 3DPW −2 to −4 mm. | Medium-High | Yes — zero-initialized corrections + gate. |
| 19 | **SMPL Human-Model Bridge** (`v52_smpl_human_model_bridge_v52.md`) | Regress SMPL params, forward body, blend regressed joints back. | WebBridge/H36M −3 to −6 mm; MPI/3DPW up to −8 mm. | High | Yes — zero-initialized blend. |
| 20 | **Implicit Neural Geometry** (`v52_implicit_neural_geometry_v52.md`) | Implicit pose manifold field + optional inner-loop camera/pose refinement. | Sparse/cross-domain −0.8 to −5 mm. | High | Yes — zero-initialized field/residual. |

## 3. Top-1 Recommendation: Uncertainty-Weighted Triangulation (UWT)

**Module:** `UncertaintyWeightedTriangulationV52` → `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py`

**Why:** It is the smallest, lowest-risk extension of the proven v45/v46/v51 stack that directly targets the next paper-stage bottleneck: *how per-view 2-D evidence is fused into a 3-D point*. v51 made reliability domain-aware; UWT makes the triangulation itself precision-aware, so the model can explicitly down-weight noisy or occluded views before the 3-D point is ever formed. Because the precision MLP and residual MLP are zero-initialized, a v51 checkpoint loads unchanged and the baseline is preserved until the new head learns.

UWT is a natural continuation of the paper narrative. It sits squarely in the **multi-view fusion and calibration** stage, consumes the same per-view feature tokens already produced by the ST transformer and v51 CDSVR, and outputs refined 3-D points that feed the existing physical-space alignment and temporal heads. It avoids the dependency, runtime, and calibration risks of the SMPL/ING/DBA proposals, the memory overhead of the video-feature and temporal-sync proposals, and the redundancy risks of stacking another reliability head. Its acceptance criteria are also simple and verifiable: at initialization the triangulation weights are uniform, and the output pose equals the baseline.

## 4. Justification

The v51 synthesis identified the cross-domain sparse-view reliability gap as the highest-priority target. v51 closed that gap by making the residual-to-reliability mapping domain-conditional. The next logical step is to question whether the *triangulation weights* themselves should be learned. Today, v45 adaptive geometry fusion and v46/v51 reliability heads influence triangulation, but they operate on top of an initial DLT that still uses hand-shaped or indirectly learned weights. UWT puts the weight learning at the point where 2-D rays are converted to 3-D points, giving the model direct gradient access to the geometry of multi-view fusion. This is more targeted than adding another attention block after the ST transformer, and it is more fundamental than a test-time or loss-only refinement because it changes the pose estimate before downstream modules see it.

UWT is also the safest high-impact candidate. It introduces a single MLP-based precision predictor, a weighted DLT, and a small residual correction — all of which can be unit-tested for identity-at-init and gradient stability. The risk register highlights weight collapse, ill-conditioned DLT gradients, and double-counting with v45/v46/v51, but each has a concrete mitigation (entropy regularization, damping/SVD, and treating UWT as the primary triangulation weight respectively). The module can be smoke-tested on the local RTX 4090 in a few hours and, if it passes, dropped into the A800 queue with minimal changes to the trainer or data loader. Compared with the SMPL bridge or implicit neural geometry proposals, UWT does not introduce new dependencies, parametric body models, or inner optimization loops; compared with the temporal-sync or video-feature proposals, it does not raise memory or clip-length concerns.

Finally, UWT is the best launchpad for the other v52 candidates. Once the triangulation weights are learned and exposed, the outlier-robust reliability head, the learned view-selection policy, and the geometry-aware attention pooling proposals can consume or refine those weights. Choosing UWT first therefore preserves optionality: it improves the shared bottleneck that many other proposals assume is already good enough.

## 5. Runner-Up Candidates for v53/v54

1. **Outlier-Robust Reliability** — A direct v51 extension that adds a Tukey/Huber M-estimator and physical cues (bone/floor) to the reliability head. Low implementation cost and strong synergy with UWT; run it once UWT has stabilized the triangulation weights.

2. **Geometry-Aware Attention Pooling (GAAP)** — Geometry-biased cross-view attention after the ST transformer. Strong paper fit and identity-at-init, but higher memory cost and risk of epipolar bias overpowering learned attention. A natural v53 after UWT has improved the triangulation signal.

3. **Physical-Space Calibration** — Jointly refines cameras and pose using physical priors. Directly closes the *multi-view fusion and calibration → physical-space alignment* loop, but camera-overfitting and SO(3) gradient risks make it better suited to a dedicated v54 ablation.

4. **Multi-Scale Geometry Fusion** — Multi-scale (joint/part/body) cross-view attention with learnable soft assignment. Good sparse-view regularization, but part-group mismatch across datasets and extra compute push it behind UWT and GAAP.

5. **Learned View Selection Policy** — Differentiable policy that selects camera subsets before triangulation. Conceptually complementary to UWT, but the hard/straight-through selection path and budget-collapse risks warrant a separate, focused v53 run.
