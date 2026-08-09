# ICRA / CVPR 2027 Publication Storyline

**Working title:** Geometry-First Multi-View 3D Human Pose: When a Strong Baseline Outperforms Complex Fusion Stacks

**Target venues:** CVPR 2027 (primary) / ICRA 2027 (secondary, robotics angle)

**Story status:** As of 2026-08-09, the best A800 result is **v25 geometry fusion at 17.17 mm val_MPJPE**, while all v31–v43 stacks remain behind (~26–37 mm). The paper therefore pivots to a **geometry-first, minimal-fusion narrative** rather than a complex attention/graph story.

---

## 1. One-sentence thesis

A carefully designed geometry-aware multi-view fusion baseline—combining ray-aware attention, learned depth triangulation, and bounded bundle adjustment—outperforms far more elaborate graph- and attention-based architectures and provides a calibrated, plug-in module for robotics and immersive video.

---

## 2. Abstract (≈150 words)

Multi-view 3D human pose estimation is the capture backbone for human-robot collaboration, sports analytics, and AR/VR. Recent work has pursued ever larger attention and graph networks, yet we show that a geometry-first baseline is substantially stronger. We present MotionFlow-MultiView v25, a calibrated multi-view fusion module that (1) represents every 2D detection as a world-space ray, (2) scores cross-view agreement with epipolar and 3D ray-intersection quality, (3) fuses per-view depth proposals into a camera-consistent 3D estimate, and (4) refines joints and cameras with a bounded bundle-adjustment block. On the WebBridge/H36M/MPI mixed benchmark, our v25 geometry-fusion model reaches **17.17 mm val_MPJPE**, outperforming deformable-attention (20.24 mm), hierarchical (21.54 mm), and the full v31–v43 stack (≥26 mm). The module is identity-initialized and warm-startable, making it a drop-in plugin for existing pipelines.

---

## 3. Core contributions

1. **Geometry as the language of fusion.** We keep triangulation, rays, and camera geometry central, rather than treating them as soft biases in a black-box network.
2. **Ray-aware cross-view attention.** Attention logits are grounded in epipolar distance and 3D ray-intersection quality, not only content similarity.
3. **Learned depth-proposal triangulation head.** Per-view depth hypotheses are scored and fused into a camera-consistent 3D joint estimate.
4. **Bounded geometry bundle adjustment (GeoBA).** A lightweight analytic refinement step with reprojection and cheirality constraints.
5. **Plug-in integration.** The block is identity at initialization and warm-starts from v18/v23 checkpoints.
6. **Empirical pivot.** Systematic A800 experiments show that v25 outperforms the v31–v43 complex stacks, suggesting that careful geometry is more valuable than added architectural complexity.

---

## 4. Supporting results as of 2026-08-09

### 4.1 Best A800 runs

| Rank | Model | Best val_MPJPE (mm) | Notes |
|------|-------|--------------------:|-------|
| 1 | **v25 geometry fusion full** | **17.17** | Best overall; reached at epoch 1 |
| 2 | v25 geometry fusion small | 18.31 | Confirms v25 is stable across scales |
| 3 | v11 IRLS | 20.06 | Strong classical baseline |
| 4 | v10 aleatoric outlier | 20.16 | Uncertainty-aware triangulation |
| 5 | v18 deformable attention | 20.24 | Prior best attention baseline |
| 6 | v12 adaptive multiscale | 20.56 | Multiscale fusion |
| 7 | v29o hierarchical n_st=3 | 21.54 | Hierarchical-only best |
| 8 | v32 combined | 26.49 | v31–v34 complex stack |
| 9 | v31 hierarchical more dropout | 26.97 | Hardened hierarchical |
| 10 | v33 uncertainty-aware tri | 27.57 | v33 component |

### 4.2 Local RTX 4090 best epoch-1 results

| Rank | Model | val_MPJPE (mm) | Notes |
|------|-------|---------------:|-------|
| 1 | v2 d128 no graph | 24.71 | Simplest local baseline |
| 2 | v2 d128 dense graph v2 | 25.19 | |
| 3 | v34 HMSP+geometry VJGN stack | 25.50 | Complex stack local peak |
| 4 | v33 combined | 25.78 | |
| 5 | v42 v36+physical+domain | 26.16 | Pending A800 confirmation |
| 6 | v36 UGIGR | 26.42 | |
| 7 | v37 SCVR | 26.94 | |

### 4.3 Key observations

- **v25 is the strongest known model.** Its 17.17 mm result is ~3 mm better than the next best completed A800 run and ~9 mm better than v31–v43 stacks.
- **Complexity has not paid off.** v31 geometry attention (33.69 mm), v31 outlier adaptive (37.87 mm), and v32/v33 combined (~26–27 mm) all trail v25.
- **Overfitting is systemic.** v25 full reached 17.17 mm at epoch 1 but later overfit to 59.14 mm, suggesting future work should emphasize epoch-1 validation, strong regularization, and early stopping.
- **Local results do not guarantee A800 performance.** v42/v43 look promising locally (26 mm) but still far above v25 on A800.

---

## 5. Method narrative for the paper

### 5.1 Pipeline

```
Multi-view video
  -> 2D keypoints + confidences
  -> Ray tokenisation
  -> Geometry-aware cross-view attention
  -> Learned depth-proposal triangulation
  -> Bounded GeoBA refinement
  -> Residual refinement (optional)
  -> 3D pose + uncertainty
```

### 5.2 Ray tokenisation

For each view v and joint j:
- Camera center `c_v = -R_v^T t_v`
- Ray direction `d_vj ∝ R_v^T K_v^{-1} [u_vj, v_vj, 1]^T`
- Ray token: `MLP([d_vj; c_v; conf_vj; z_emb])`

### 5.3 Geometry-aware cross-view attention

The attention logit is a sum of three terms:
- **Content logit:** scaled dot-product on projected ray features.
- **Epipolar logit:** distance from the query ray to the epipolar line of the key ray.
- **Ray-intersection logit:** `-ray_dist / σ_d - (1 - cosθ) / σ_θ`.

Masked views are excluded; the output is a residual added to feature tokens.

### 5.4 Learned depth-proposal triangulation head

- Sample `n_ray_samples` depths along each ray.
- Project to 3D candidates `X_vj^k = c_v + z_k d_vj`.
- Score candidates with cross-view attention conditioned on ray-intersection and epipolar quality.
- Aggregate per joint: `X_j = Σ_v w_v X_vj`.

### 5.5 Bounded geometry bundle adjustment

- **Structure step:** 1–2 damped Levenberg-Marquardt steps on reprojection error, clamped by `max_point_update_m`.
- **Camera step:** lightweight MLP predicts bounded camera correction from reprojection residual and ray-intersection quality; initialized to identity.
- **Geometry losses:** reprojection, epipolar, cheirality, depth consistency.

---

## 6. Experiments to report

### 6.1 Datasets and metrics

- **Human3.6M:** 4 views, 17 joints; train S1, validate S5/Act2.
- **MPI-INF-3DHP:** 14 views, 28 joints; train S1/S3, validate S2/Seq1.
- **WebBridge cross-dataset benchmark:** mixed training with H36M/MPI/WebBridge.
- Metrics: MPJPE, PA-MPJPE, PCK@50/100/150, AUC.

### 6.2 Main results table (target)

| Dataset | Model | MPJPE (mm) | PA-MPJPE (mm) |
|---------|-------|------------:|----------------:|
| MPI-INF-3DHP | Raw DLT | 25.21 | 24.08 |
| MPI-INF-3DHP | v18 deformable attention | 20.24 | TBD |
| MPI-INF-3DHP | v25 geometry fusion (d=128) | **17.17** | TBD |
| H36M | v25 geometry fusion (d=128) | TBD | TBD |

### 6.3 Ablation table

| Ablation | val_MPJPE (mm) | Question answered |
|----------|---------------:|-------------------|
| Raw DLT | ~25 | Geometric baseline |
| v18 cross-view residual + PP | 20.24 | Best non-geometry baseline |
| v25 w/o geometry attention | TBD | Is ray-intersection attention necessary? |
| v25 w/o learned depth triangulation | TBD | Value of depth-proposal head |
| v25 w/o GeoBA | TBD | Value of analytic refinement |
| v25 full | 17.17 | Geometry-first fusion |

### 6.4 Robustness tests

- 2D keypoint Gaussian noise.
- Random joint occlusion.
- Random view dropout (variable-view inference, k = 2..14).
- Controlled camera perturbations (rotation, translation, focal length, principal point).

---

## 7. Figures and tables to prepare

1. **Architecture diagram:** 2D keypoints → ray tokenisation → geometry-aware cross-view attention → learned depth-proposal triangulation → GeoBA → 3D pose + uncertainty.
2. **Main results bar chart:** DLT, v18, v25 on MPI-INF-3DHP and H36M.
3. **Ablation chart:** v25 full vs. no-attn / no-depth / no-GeoBA.
4. **Calibration robustness heatmap:** MPJPE under rot/trans/focal/PP perturbations.
5. **Variable-view curve:** MPJPE@k for k = 2..14.
6. **Model complexity vs. accuracy plot:** show v25 (simple, low error) vs. v31–v43 (complex, higher error).
7. **Failure-case heatmaps:** per-joint/per-view error and ray-intersection quality.

---

## 8. Publication risks and mitigations

| Risk | Evidence | Mitigation |
|------|----------|------------|
| v25 overfits after epoch 1 | 17.17 mm → 59.14 mm on A800 | Early stopping, SWA/EMA, weight decay, dropout |
| v42/v43 A800 runs may still beat v25 | Local v42 at 26.16 mm is far above v25 | Wait for A800 results before finalizing storyline |
| Cross-dataset transfer unknown | WebBridge mixed runs pending | Run v25 all-train baseline on A800 |
| Comparison to literature unclear | Need recent SOTA numbers | Audit CVPR/ICCV/ECCV 2025–2026 papers |
| Camera-perturbation robustness unverified | Planned ablations not all run | Add robustness harness before submission |

---

## 9. Recommended next steps

1. **Await A800 v25/v42/v43 results** (issue #154) before locking the paper angle.
2. **Lock the v44 architecture:** if v25 all-train remains best, build v44 as v25 + physical loss + domain weights + SWA.
3. **Run the full v25 ablation smoke** on RTX 4090: verify that geometry attention, depth-proposal head, and GeoBA each contribute.
4. **Generate the main figures** (architecture, results, ablation, robustness) once v25 numbers are stable.
5. **Draft the introduction around the “geometry-first” pivot**, explicitly contrasting v25 with v31–v43.
6. **Survey CVPR/ICCV/ECCV 2025–2026 multi-view pose papers** to position the 17.17 mm result against recent SOTA.
7. **Prepare a 2-minute supplementary video** showing ray-aware attention weights under occlusion and view dropout.

---

## 10. Submission timeline (tentative)

| Milestone | Target date | Deliverable |
|-----------|------------:|-------------|
| v44 A800 results | 2026-08-15 | Decide final architecture |
| v25 full ablation + robustness | 2026-08-22 | Tables/figures ready |
| Literature comparison | 2026-08-29 | Related work section complete |
| Paper first draft | 2026-09-15 | Full ICRA/CVPR draft |
| Supplementary material | 2026-09-30 | Video + appendix |
| CVPR 2027 deadline | ~November 2026 | Submit |

---

## 11. Conclusion

The emerging ICRA/CVPR 2027 story is:

> **Multi-view 3D human pose estimation does not need heavier attention or graph stacks; it needs geometry to be the core language of fusion.** A simple but principled geometry-aware baseline (v25) currently outperforms every complex variant we have built, and with modest additions—physical priors, domain-balanced training, and stronger regularization—it can form the basis of a compelling paper.

This storyline is conditional on the pending A800 v25/v42/v43 results confirming the current trend.
