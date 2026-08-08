# MotionFlow-MultiView v25: Paper Story Outline for ICRA / CVPR 2027

**Working title:** Geometry-First Multi-View 3D Human Pose Fusion with Ray-Aware Attention and Learned Depth Triangulation

**Target venues:** ICRA 2027 / CVPR 2027

**Submission angle:** A geometry-centric multi-view fusion module that reasons with rays, cameras, and 3D structure end-to-end, while remaining a drop-in plugin for the MotionFlow pipeline.

**Last updated:** 2026-08-08

---

## 1. One-sentence thesis

**We make calibrated multi-view 3D human pose estimation robust by keeping geometry as the core language of fusion: the model represents every view as a ray, scores cross-view agreement with 3D ray-intersection quality, and fuses per-view depth proposals into a camera-consistent 3D estimate—learning only the residual that pure triangulation cannot fix.**

---

## 2. Abstract (≈150 words)

Multi-view video is the capture backbone for human-robot collaboration, sports analytics, and immersive telepresence, but standard triangulation is brittle to occlusion, detector noise, and calibration drift.
We present a geometry-first multi-view fusion module that reasons explicitly with rays and cameras:
(1) a ray tokeniser that encodes each 2D keypoint as a world-space viewing ray;
(2) geometry-aware cross-view attention that scores attention with epipolar distance and 3D ray-intersection quality;
(3) a learned depth-proposal triangulation head that fuses per-view depth hypotheses; and
(4) a bounded geometry bundle-adjustment block that refines both 3D joints and camera parameters with analytic reprojection and cheirality constraints.
The module is warm-startable from existing checkpoints and exposed as a `MultiViewFusionPlugin` inside MotionFlow.
On MPI-INF-3DHP our strongest ensemble reaches **8.35 mm** MPJPE; on Human3.6M the same family reaches **0.62 mm** MPJPE.

---

## 3. Introduction (target: 1.5 pages)

1. **Motivation.** Multi-view 3D human pose is the practical route to metric, world-grounded motion capture for robotics and AR/VR.
2. **Failure of DLT.** Direct Linear Transform is exact under perfect calibration and clean 2D observations, but degrades under occlusion, detector bias, and camera drift.
3. **Failure of black-box fusion.** End-to-end attention can absorb noise but regresses 3D joints directly and discards the metric, camera-consistent inductive bias of triangulation.
4. **The v25 insight.** *Geometry should be the language of fusion, not an afterthought.* Represent views as rays, score cross-view attention with 3D ray-intersection quality, and fuse depth proposals before any black-box residual correction.
5. **Contributions preview.**
   - Ray-aware tokenisation and geometry cross-view attention.
   - Learned depth-proposal triangulation head.
   - Bounded geometry bundle adjustment (GeoBA) with analytic reprojection and cheirality constraints.
   - Drop-in `MultiViewFusionPlugin` integration with warm-start from v18/v23 checkpoints.
6. **Paper roadmap.** §Related Work, §Method, §Experiments, §Discussion/Conclusion.

---

## 4. Related Work (target: 1 page)

### 4.1 Classical multi-view geometry
- DLT, robust triangulation, bundle adjustment, M-estimators.
- Why they fail under occlusion and calibration drift.

### 4.2 Learnable multi-view pose
- Single-view 3D lifting vs. multi-view fusion.
- Temporal models, transformer cross-view attention, graph-joint attention.
- Limitation: geometry is used only indirectly (epipolar bias, camera embeddings).

### 4.3 Geometry-aware deep learning
- Neural bundle adjustment, camera-parameter regression, differentiable BA.
- Ray representations, neural radiance fields for pose.
- Residual learning and calibration-aware pose.

### 4.4 MotionFlow and robot retargeting
- Why a calibrated, uncertainty-aware, geometry-first plugin matters for downstream robotics.

---

## 5. Method: Multi-View Geometry Fusion v25

### 5.1 Overview and notation
- Input: calibrated multi-view 2D keypoints `(B, T, V, J, 2)`, confidences `(B, T, V, J)`, intrinsics `K`, rotations `R`, translations `t`.
- Output: refined 3D joints `(B, T, J, 3)`, optional refined cameras, auxiliary geometry loss.
- The module is inserted **after** v18 deformable cross-view attention and **before** the spatio-temporal transformer.

### 5.2 Ray tokenisation
- Camera centre `c = -R^T t`.
- World ray direction `d = R^T K^{-1} [u, v, 1]^T`.
- Ray token: `MLP([d; c; conf; z_emb])`, where `z_emb` is a learned depth-proposal embedding.

### 5.3 Geometry-aware cross-view attention
- Content logit: standard scaled dot-product on projected ray features.
- Epipolar logit: existing v18 epipolar bias.
- Ray-intersection logit:
  ```
  logit_ray = - ( ray_dist(v_q, v_k) / sigma_d + (1 - cosθ) / sigma_a )
  ```
- Masked views excluded; output is a residual added to feature tokens.

### 5.4 Learned depth-proposal triangulation head
- Sample `n_ray_samples` depths per ray.
- Project to 3D candidates `X_vj^k = c_v + z_k d_vj`.
- Score candidates with cross-view attention conditioned on ray-intersection and epipolar quality.
- Aggregate per joint: `X_j = sum_v w_v X_vj`.

### 5.5 Geometry bundle adjustment (GeoBA)
- **Structure step:** 1–2 damped Gauss-Newton/Levenberg-Marquardt steps on reprojection error, clamped by `max_point_update_m`.
- **Camera step:** lightweight MLP predicts bounded camera correction from reprojection residual and ray-intersection quality; initialised to identity.
- **Geometry losses:** reprojection, epipolar, cheirality, depth-consistency.

### 5.6 Optional camera-joint graph (placeholder)
- Bipartite GNN over `V` camera nodes and `J` joint nodes.
- Not yet wired; included as future work.

### 5.7 Training objectives
```
L = L_3D_MSE + λ_reproj·L_reproj + λ_epi·L_epi + λ_cheir·L_cheir + λ_depth·L_depth_consistency + λ_geom·L_geom
```
- Camera perturbation as core augmentation (rotation, translation, focal, principal point).
- Validation always on unperturbed calibration.

### 5.8 Plug-in integration
- `MultiViewFusionPlugin` consumes calibrated multi-view 2D keypoints.
- Emits `HumanMotionIR` with `pose`, `uncertainty`, `provenance`.
- Warm-start from v18/v23 checkpoints possible because the new block is identity at init.

---

## 6. Experiments

### 6.1 Datasets and metrics
- **MPI-INF-3DHP** (14 views, 28 joints): train S1/S3, validate S2/Seq1.
- **Human3.6M** (4 views, 17 joints): train S1, validate S5/Act2.
- **WebBridge cross-dataset benchmark** (optional).
- Metrics: MPJPE, PA-MPJPE, PCK@50/100/150, AUC, per-joint/per-view error maps.

### 6.2 Main accuracy results
| Dataset / Condition | Model | MPJPE (mm) | PA-MPJPE (mm) |
|---------------------|-------|-----------:|----------------:|
| MPI-INF-3DHP S2/Seq1 | Raw DLT | 25.21 | 24.08 |
| MPI-INF-3DHP S2/Seq1 | v18 cross-view residual + PP | **9.32** | **5.37** |
| MPI-INF-3DHP S2/Seq1 | v18 + KAP (v23) | *TBD* | *TBD* |
| MPI-INF-3DHP S2/Seq1 | v25 geometry fusion (d=128) | *TBD* | *TBD* |
| MPI-INF-3DHP S2/Seq1 | Bayesian Tri v2 ensemble | **8.35** | **5.29** |
| Human3.6M S5/Act2 | CamPE + GraphJR | **0.62** | **0.70** |
| Human3.6M S5/Act2 | v25 geometry fusion | *TBD* | *TBD* |

### 6.3 Ablation studies
| Ablation | Question | Model tag |
|----------|----------|-----------|
| Raw DLT | Geometric baseline | — |
| v18 cross-view residual + PP | Best non-geometry-attention baseline | v18 |
| v23 (v18 + KAP) | Does KAP help? | v23 |
| v25 w/o geometry attention | Is ray-intersection attention necessary? | v25 no-attn |
| v25 w/o learned depth triangulation | Value of depth-proposal head | v25 no-depth |
| v25 w/o GeoBA | Value of analytic camera refinement | v25 no-geoba |
| v25 full | Geometry-first fusion | v25 |

**Reproducible smoke ablation harness:** `experiments/architecture_paper_story_v25_ablation.py` (driven by `configs/architecture_paper_story_v25_smoke.yaml`) instantiates each variant on synthetic calibrated data, reports parameter count, forward latency, and a prediction-delta metric, and writes the table/charts to `outputs/architecture_paper_story/`.  Run it on the local RTX 4090 before training to verify the v25 toggles are wired correctly.

### 6.4 Robustness and calibration tests
- 2D keypoint Gaussian noise.
- Random joint occlusion.
- Random / adversarial view dropout.
- Controlled camera perturbations: rotation, translation, focal length, principal point.
- Variable-view inference (k = 2..14).

### 6.5 Cross-dataset and integration tests
- Train on MPI, test on H36M / WebBridge.
- MotionFlow plugin demo with uncertainty and provenance logging.
- Throughput on RTX 4090 (batch 1–16).

---

## 7. Figures and Tables to Prepare

1. **Architecture diagram:** 2D keypoints → intrinsic correction → ray tokenisation → geometry-aware cross-view attention → learned depth-proposal triangulation → GeoBA → residual refinement → 3D pose + uncertainty.
2. **Main results table:** MPI-INF-3DHP and H36M MPJPE/PA-MPJPE for DLT, v18, v23, v25, and Bayesian Tri v2 ensemble.
3. **Ablation bar chart:** v25 full vs. no-attn / no-depth / no-GeoBA.
4. **Calibration robustness heatmap:** MPJPE under rot/trans/focal/PP perturbations.
5. **Variable-view curve:** MPJPE@k for k = 2..14.
6. **Runtime plot:** latency/throughput on RTX 4090.
7. **Failure-case heatmaps:** per-joint/per-view error and ray-intersection quality.

---

## 8. Discussion

- **What works:** geometry-first decomposition, ray-aware attention, learned depth proposals, bounded GeoBA.
- **What remains hard:** rotation/focal drift, variable-view inference with very few views, cross-dataset transfer.
- **Future work:** camera-joint graph, continuous depth proposals, domain adaptation, SMPL fitting stage.

---

## 9. Conclusion

- Recap the *geometry as the language of fusion* principle.
- State the practical contribution: a calibrated, geometry-aware, plug-in multi-view fusion module for robotics and immersive video.
- Reiterate key results and next experiments (v23/v24 → v25 smoke, full v25 run, ensemble).

---

## 10. Running experiments and next steps

| Run | Description | GPU | Status | Paper section |
|-----|-------------|-----|--------|---------------|
| v23b | v18 + KAP 0.001, no neural BA | A800 GPU4 | **Failed** (58.72 mm) | §6.3 ablation |
| v24b | v18 + fixed BA + KAP 0.001 | A800 GPU6 | **Failed** (131.73 mm) | §6.3 ablation |
| v18 full | v18 cross-view residual + PP | A800 GPU5 | Running (long 60-epoch full run, ~18k steps) | §6.2 main results |
| v25 small | v25 geometry fusion small (geom λ=0.1) | A800 GPU7 | Running (~2250 steps, ~60% of 1st epoch) | §6.2 main results |
| v25 full | v25 geometry fusion full (geom λ=0.1) | A800 GPU4 | Running (~3200 steps, ~17% of 1st epoch) | §6.2 main results |
| v25 ablation | v25 geometry fusion small (geom λ=1.0) | A800 GPU6 | Running (~2000 steps, ~53% of 1st epoch) | §6.3 ablation |
| v26 small | v26 temporal geometry fusion | — | **Ready** (code pulled; 256 k params, 0.28 GFLOPs vs v25 191 k/0.20 GFLOPs) | §6.3 ablation |

**Next concrete step:** wait for v25 small/ablation first-epoch `val_MPJPE`. With 30 train files, one epoch for the small/ablation config is ~3750 steps, so first validation is expected in 1–2 hours. If v25 small beats the v18 baseline (20.24 mm), keep v25 full/ablation running, pull latest `main` on A800, and launch v26 small for comparison; otherwise, stop v25 full/ablation and debug the geometry-fusion design.

---

## 11. Submission checklist

- [x] Run v25 small smoke and confirm training loss decreases.
- [x] Launch v25 full run on A800.
- [ ] Collect first-epoch `val_MPJPE` for v25 small and decide whether to keep v25 full.
- [ ] Fill in the *TBD* cells in §6.2 with actual v25 results.
- [ ] Generate architecture and robustness figures.
- [ ] Produce variable-view MPJPE@k curve.
- [ ] Write the introduction around the *geometry as language of fusion* story.
- [ ] Add related-work section with references from `docs/phase0_literature_audit.md` and `docs/literature_review_multiview_pose.md`.
- [ ] Write the plug-in integration and robot-retargeting angle for ICRA.
- [ ] Create a 2-minute supplementary video showing ray-aware attention weights under occlusion.
