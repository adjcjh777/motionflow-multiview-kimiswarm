# v53 Design Space Synthesis

Synthesized from the 20 design proposals in `docs/swarm_iter27/proposals/` and their risk reports in `docs/swarm_iter27/reports/`. The project is at the v52 Uncertainty-Weighted Triangulation milestone: v52 makes the triangulation step precision-aware and is currently being smoke-tested. v53 should build on that warm-started backbone and attack the next bottleneck in the paper pipeline — calibrating the triangulated 3-D pose against physical-space invariants before downstream temporal/refinement heads.

## 1. Executive Summary

v52 gave the model learnable triangulation weights. The remaining gap is that the pose returned by v52 is still a purely geometric DLT output; it is not yet calibrated against the physical world (floor plane, bone-length skeleton, camera-pose consistency). v53 should therefore add a **physical-space calibration** stage that is identity-at-init, consumes the v52 weights as a robustness signal, and refines the 3-D pose before physical-space alignment and temporal heads.

The 20 proposals fall into five natural clusters:

1. **Physical / calibration priors** (physical-space calibration, bone-length-aware fusion, differentiable bundle adjustment, camera noise correction, SMPL human-model bridge, implicit neural geometry).
2. **Triangulation / view weighting refinements** (learned view selection policy, multi-scale geometry fusion, outlier-robust reliability, geometry-aware attention pooling).
3. **Domain / data-level improvements** (adaptive LR per domain, cross-dataset domain bridge, domain-conditional normalization, self-supervised multi-view pretraining).
4. **Temporal / consistency** (multi-view temporal sync, temporal consistency loss, probabilistic pose forecasting, video feature extractor).
5. **Test-time / augmentation** (test-time self-refinement, reliability-guided pose mixup).

All 20 are identity-at-init by construction; the ranking below orders them by expected sparse/full-view MPJPE impact per unit risk, with heavy preference for paper-aligned, low-risk candidates.

## 2. Ranking Table of the 20 v53 Proposals

| Rank | Proposal (file) | Key idea | Expected MPJPE impact | Risk | Warm-start / identity-at-init |
|---:|---|---|---|---|---|
| 1 | **Physical-Space Calibration** (`v53_physical_space_calibration_v53.md`) | Calibrate the v52 pose against floor plane and canonical bone lengths using v52 weights; gated residual refiner. | Identity `< 0.1 mm`; smoke `−2 to −5 mm`; full `1–3%` gain, larger on `@2/3`. | Medium | Yes — final MLP zero-init and gate logit `−6.0`. |
| 2 | **Temporal Consistency Loss** (`v53_temporal_consistency_loss_v53.md`) | Use v52 precision weights to modulate temporal smoothness, velocity, and bone-length consistency losses. | Smoke `≤ 0.5 mm`; `@full 0.4–1.0 mm`; `@2/3 1.0–2.5 mm`. | Medium | Yes — loss weight ramped from zero; no new forward module. |
| 3 | **Learned View Selection Policy** (`v53_learned_view_selection_policy_v53.md`) | Differentiable Gumbel-softmax policy on v52 weights that selects camera subsets before weighted DLT. | Full `0.2–0.8 mm`; sparse `0.8–2.0 mm`. | Medium | Yes — gate `sigmoid(−6) ≈ 0`, so `X^(1)=X^(0)`. |
| 4 | **Multi-Scale Geometry Fusion** (`v53_multi_scale_geometry_fusion_v53.md`) | Fuse v52 weights across joint/limb/body scales with cross-view attention and scale-aware 3-D correction. | Full `0.5–1.0 mm`; `@2 2–4 mm`; `@3 1–2 mm`. | Medium | Yes — zero-init output layers and residual gate. |
| 5 | **Self-Supervised Multi-View Pretraining** (`v53_self_supervised_multiview_pretraining_v53.md`) | Auxiliary masked-view consistency loss asking v52 UWT to reconstruct full-view triangulation. | H36M/MPI `0.5–1.2 mm`; sparse `1.5–3.5 mm`; cross-domain `2–4 mm`. | Medium | Yes — no forward-path change; zero-init projections. |
| 6 | **Multi-View Temporal Sync** (`v53_multi_view_temporal_sync_v53.md`) | Predict per-view sub-frame temporal offsets and warp/fuse trajectories using v52 weights. | Full `0.5–1.0 mm`; sparse `1.5–3.0 mm`. | Medium | Yes — residual gate `=0`, so `X^out=X^U`. |
| 7 | **Probabilistic Pose Forecasting** (`v53_probabilistic_pose_forecasting_v53.md`) | Forecast per-joint Gaussian pose distribution from v52-weighted history and fuse via uncertainty-gated smoothing. | Local `−0.8 to −1.5 mm`; full `−0.3 to −0.8 mm`; `@2 −1.0 to −2.0 mm`. | Medium | Yes — final layers and gate zero-initialized. |
| 8 | **Test-Time Self-Refinement** (`v53_test_time_self_refinement_v53.md`) | Iteratively refine the v52 pose at test time using reprojection residuals and a skeleton graph network. | Smoke no regression; medium `1–2 mm`; full `2–4 mm`; cross-domain `3–5 mm`. | Medium | Yes — correction/gate heads zero-initialized. |
| 9 | **Outlier-Robust Reliability** (`v53_outlier_robust_reliability_v53.md`) | Refine v52 weights with a learned robust kernel over reprojection, epipolar, temporal, and physical residuals. | Smoke `−0.5 to −1.5 mm`; full `−1.0 to −2.5 mm`; larger on `@2/3`. | Medium-High | Yes — final projections zero-init; gate `−6.0`. |
| 10 | **Bone-Length-Aware Fusion** (`v53_bone_length_aware_fusion_v53.md`) | Add a bone-length-aware fusion stage after v52 that learns per-bone corrections before downstream heads. | Sparse `−1.5 to −3.5 mm`; full `−0.3 to −1.0 mm`. | Medium-High | Yes — final projections zero-init; residual gate `0.0`. |
| 11 | **Reliability-Guided Pose Mixup** (`v53_reliability_guided_pose_mixup_v53.md`) | Generate an ensemble of v52-weighted triangulation hypotheses, score by geometric consistency, fuse best. | Sparse `−1.0 to −3.0 mm`; full `−0.2 to −0.8 mm`. | Medium | Yes — candidate multipliers and residual gate zero-init. |
| 12 | **Adaptive LR per Domain** (`v53_adaptive_lr_per_domain_v53.md`) | Rescale optimizer step per domain using v52 uncertainty/precision as a live difficulty signal. | Smoke `≤0.5 mm`; medium `0.5–2 mm`; full `1–3 mm` cross-domain. | Medium | Yes — only affects optimizer step; `α_d ≡ 1` during warmup. |
| 13 | **Domain-Conditional Normalization** (`v53_domain_conditional_normalization_v53.md`) | Domain-conditionally recalibrate v52 UWT weights and triangulated pose before physical-space alignment. | Mixed-domain `0.5–1.2 mm`; sparse improves; warm-start `<0.1 mm`. | Medium | Yes — zero-init final layers and residual gate. |
| 14 | **Geometry-Aware Attention Pooling** (`v53_geometry_aware_attention_pooling_v53.md`) | Pool per-view feature tokens with pairwise camera-geometry embeddings and gated residual pose update. | Smoke `0.5–1.0 mm`; full `0.3–0.8 mm`; largest on `@2/3`. | Medium-High | Yes — zero-init output projection and pose MLP. |
| 15 | **Video Feature Extractor** (`v53_video_feature_extractor_v53.md`) | Extract causal spatiotemporal video features before v52 UWT so triangulation weights exploit motion dynamics. | Smoke within `0.5 mm`; full `0.5–1.5 mm`; `@2` up to `2 mm`. | Medium-High | Yes — final output projection zero-init; gate `g=0`. |
| 16 | **Cross-Dataset Domain Bridge** (`v53_cross_dataset_domain_bridge_v53.md`) | Recalibrate v52 pose into a domain-invariant canonical skeleton via FiLM and cross-domain pose-prototype attention. | Source-only `±0.5 mm`; H36M→MPI `−2 to −4 mm`; H36M→3DPW `−4 to −7 mm`. | Medium-High | Yes — zero-init residual MLP and gate. |
| 17 | **Differentiable Bundle Adjustment** (`v53_differentiable_bundle_adjustment_v53.md`) | Lightweight differentiable Levenberg-Marquardt block after v52 to jointly refine pose and cameras. | Sparse `@2/3 1–3 mm`; cross-domain/WebBridge/3DPW `2–4 mm`. | Medium-High | Yes — all correction MLP final layers zero-init; gate `=0`. |
| 18 | **SMPL Human-Model Bridge** (`v53_smpl_human_model_bridge_v53.md`) | Convert v52 skeleton to SMPL body and blend SMPL-aligned joints back as a gated residual. | Full `−1 to −2.5 mm`; sparse `−2 to −4 mm`. | Medium-High | Yes — blend gate closed and residual gate `=0` at init. |
| 19 | **Camera Noise Correction** (`v53_camera_noise_correction_v53.md`) | Learn gated per-view affine 2-D correction in normalized image coordinates before v52 triangulation. | Conservative `−1.0 to −2.0 mm`; optimistic `−2.5 to −4.0 mm` on 3DPW. | Medium-High | Yes — `θ_v=0`, gate `g_v=0` at init. |
| 20 | **Implicit Neural Geometry** (`v53_implicit_neural_geometry_v53.md`) | Refine v52 pose with a ray-conditioned implicit neural geometry layer using calibrated rays and features. | Full `−0.5 to −1.0 mm`; sparse `−1.0 to −2.0 mm`; 3DPW `−1 to −3 mm`. | Medium-High | Yes — MLP final layers zero-init; `λ=0` at init. |

## 3. Top-1 Recommendation: Physical-Space Calibration (PSC)

**Module:** `PhysicalSpaceCalibrationV53` → `motionflow_mv/fusion/physical_space_calibration_v53.py`

**Why:** v52 made triangulation precision-aware; the next bottleneck in the paper narrative is **physical-space calibration** — ensuring that the triangulated 3-D pose respects real-world invariants such as the ground plane and the subject’s canonical bone lengths. PSC is the lowest-risk, most paper-aligned candidate. It consumes the v52 triangulation weights as a robustness signal, applies a gated residual correction, and is fully identity-at-init, so it can be smoke-tested on the local RTX 4090 by warm-starting from any v52 checkpoint.

PSC closes the loop between the *multi-view fusion and calibration* stage and the *physical-space alignment* stage without introducing new parametric body models, inner optimization loops, or test-time-only changes. It also leaves v46/v47/v48/v49/v50/v51 downstream heads untouched, minimizing blast radius and preserving the option to stack other v53 candidates later.

## 4. Justification

The v52 synthesis identified triangulation precision as the highest-priority target. v52 addressed it by learning per-view per-joint precision weights. The next logical step is to use those weights to **calibrate the resulting 3-D pose against physical invariants**. PSC does exactly that: a floor-calibration head removes foot-floor penetration, a bone-length calibration head regularizes the skeleton toward learned canonical lengths, and a gated residual refiner fuses these cues into a bounded 3-D correction. Because both the floor/bone losses and the residual MLP are zero-initialized and gated, a v52 checkpoint loads unchanged and the baseline is preserved until the new heads learn.

PSC is the safest high-impact candidate among the v53 proposals. It introduces a single residual MLP, learnable canonical bone-length parameters, and three auxiliary losses — all of which can be unit-tested for identity-at-init and gradient stability. The risk register highlights floor-plane violations, canonical-length conflicts, identity-at-init failure, overfitting, and double-counting with v28/v31 physical losses, but each has a concrete mitigation (gated/optional heads, per-domain canonical skeletons, zero-initialized final layers, small loss weights, and explicit ordering/ablation). The module can be smoke-tested on the local RTX 4090 in a few hours and, if it passes, dropped into the A800 queue with minimal trainer/data-loader changes.

Finally, PSC is the best launchpad for the other v53 candidates. Once the pose is physically calibrated, the bone-length-aware fusion, outlier-robust reliability, and temporal-consistency proposals have a cleaner signal to work from. Choosing PSC first therefore preserves optionality while directly advancing the paper narrative.

## 5. Runner-Up Candidates for v54/v55

1. **Temporal Consistency Loss** — A loss-only extension that uses v52 precision to modulate temporal smoothness and bone-length consistency. Lowest implementation cost and strong synergy with PSC; run it once PSC has stabilized the physical pose.

2. **Learned View Selection Policy** — Differentiable Gumbel-softmax policy on top of v52 weights. Complementary to PSC because better view selection improves both geometric triangulation and physical calibration, but the straight-through selection path and budget-collapse risks warrant a focused v54 run.

3. **Multi-Scale Geometry Fusion** — Multi-scale (joint/limb/body) cross-view attention over v52 weights. Good sparse-view regularization, but part-group mismatch and extra compute push it behind PSC and the temporal loss.

4. **Multi-View Temporal Sync** — Per-view sub-frame temporal offset prediction and warping. Strong paper fit for temporal-motion sequences, but temporal drift is a smaller bottleneck right now than the missing physical calibration stage.

5. **Differentiable Bundle Adjustment** — Jointly refines pose and cameras after v52. It directly upgrades the *calibration* part of the pipeline, but the LM/SO(3) gradient and runtime risks make it better suited to a dedicated v55 ablation after PSC is proven.
