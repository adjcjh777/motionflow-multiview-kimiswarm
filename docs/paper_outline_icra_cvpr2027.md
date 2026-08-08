# MotionFlow-MultiView: Paper Outline for ICRA / CVPR 2027

**Working title:** Geometry-First Multi-View 3D Human Pose Estimation with Bayesian Triangulation, Adaptive Refinement, and Intrinsic Self-Correction

**Target venues:** ICRA 2027 / CVPR 2027

**Submission angle:** A compact, plug-in multi-view fusion module that keeps triangulation as the geometric predictor and learns only the structured residual error (keypoint uncertainty, calibration drift, skeleton bias). Strong robot-relevant motivation: world-metric, calibrated, robust to occlusion and camera drift.

---

## 1. Abstract ( 150 words)

- Problem: multi-view 3D human pose is the capture backbone for human-robot collaboration, sports analytics, and robot policy training, but standard DLT triangulation is brittle to occlusion, detector noise, and calibration drift.
- Method: a geometry-first pipeline that (a) predicts per-view anisotropic 2D covariances for Bayesian precision-weighted triangulation, (b) refines with differentiable adaptive Gauss-Newton steps, (c) corrects principal-point/focal drift via a lightweight intrinsic head, and (d) adds a small residual MLP that operates on the camera-consistent estimate.
- Results: on MPI-INF-3DHP the best ensemble reaches **8.35 mm** MPJPE; on Human3.6M the same architecture reaches **5.24 mm** MPJPE.
- Practical hook: exposed as a `MultiViewFusionPlugin` in MotionFlow with explicit uncertainty and provenance; runs at 12–195 clips/s on a single RTX 4090.

---

## 2. Introduction

1. **Motivation.** Multi-view video is the most practical way to get metric, world-grounded human pose; use cases in robotics, AR/VR, and sports.
2. **Failure of the standard pipeline.** DLT is exact under ideal assumptions but collapses under real-world occlusion, detector bias, and calibration drift.
3. **Failure of pure learning.** End-to-end fusion regresses joints directly and discards the metric/camera-consistent inductive bias of triangulation.
4. **Thesis statement.** *Triangulate first, then learn only the residual.* This decomposition lets a compact model fix what geometry cannot.
5. **Contributions preview.**
   - Bayesian precision-weighted triangulation with anisotropic covariances.
   - Adaptive Gauss-Newton refinement with learned per-joint damping.
   - Self-calibrating intrinsic correction (principal point / focal length).
   - Small residual MLP + MotionFlow plugin interface.
6. **Paper roadmap.** Briefly introduce §3 (related work), §4 (method), §5 (experiments), §6 (discussion/conclusion).

---

## 3. Related Work

### 3.1 Classical multi-view triangulation
- DLT, triangulation from rays, robust M-estimators, bundle adjustment.
- Why they fail under occlusion and calibration drift.

### 3.2 Learnable multi-view pose estimation
- Single-view 3D lifting vs. multi-view fusion.
- Temporal models, transformer-based cross-view attention, graph-joint attention.
- Limitations: black-box fusion, no explicit camera/uncertainty model.

### 3.3 Uncertainty, robustness, and calibration-aware pose
- Probabilistic triangulation, epipolar constraints, camera-parameter conditioning.
- Principal-point / focal-length correction in the wild.
- Residual learning for pose refinement.

### 3.4 MotionFlow and robot retargeting pipelines
- Why a calibrated, uncertainty-aware plugin matters for downstream robotics.

---

## 4. Method: MotionFlow-MultiView Fusion

### 4.1 Overview and notation
- Input: calibrated multi-view 2D keypoints `(V, J, 2)`, confidences, camera intrinsics/extrinsics.
- Output: 3D joints `(J, 3)`, per-view uncertainty weights, plugin provenance.

### 4.2 Intrinsic self-correction head
- Predict per-view principal-point offset (and optional focal scale) from the 2D/confidence pattern.
- Supervised with the inverse of the training perturbation.
- Bounded output, dataset-specific ranges.

### 4.3 Ray-aware cross-view spatio-temporal attention
- Embed 2D points as camera rays.
- Transformer jointly over `(time, view)` for each joint.
- Produces context features for weighting and residual correction.

### 4.4 Bayesian precision-weighted triangulation
- Anisotropic covariance head: Cholesky factor of 2×2 image-space covariance per view/joint.
- Precision = inverse determinant; DLT weight = confidence × precision × visibility.
- Differentiable weighted DLT.

### 4.5 Adaptive Gauss-Newton refinement
- 1–2 differentiable Gauss-Newton steps in world space.
- Learned per-joint damping factor.
- Jacobian derived from the pinhole projection equations.

### 4.6 Residual refinement head
- Small MLP: `X = X_gn + MLP([pool(feat), X_gn])`.
- Learns structured biases (detector bias, mild drift, skeleton prior).

### 4.7 Training objectives
```
L = L_3D_MSE + λ_epipolar·L_epipolar + λ_pp·L_pp_offset + λ_reproj·L_reproj
```
- Camera perturbation as core augmentation (rotation, translation, focal, principal point).
- Validation always on unperturbed calibration.

### 4.8 Plug-in integration
- `MultiViewFusionPlugin` inside MotionFlow.
- Emits `HumanMotionIR` with pose, uncertainty, provenance.

---

## 5. Experiments

### 5.1 Datasets and metrics
- **MPI-INF-3DHP** (14 views, 28 joints): train on S1/S3, validate on S2/Seq1.
- **Human3.6M** (4 views, 17 joints): train on S1, validate on S5/Act2.
- Optional: 3DPW, AIST++, WebBridge cross-dataset benchmark.
- Metrics: MPJPE, PA-MPJPE, PCK@50/100/150, AUC, per-joint/per-view error maps.

### 5.2 Main accuracy results
- MPI-INF-3DHP S2/Seq1: best single model and ensemble MPJPE/PA-MPJPE.
- Human3.6M S5/Act2: MPJPE/PA-MPJPE.
- Comparison with raw DLT, temporal ray-attention, and full model.

### 5.3 Ablation studies
| Ablation | Question | Planned model tag |
|----------|----------|-----------------|
| Raw DLT | Geometric baseline | — |
| Temporal ray-attention, no residual | Value of learned weights alone | v11 / baseline |
| Cross-view residual + PP | Value of residual head + principal-point correction | v18 |
| v18 + Kinematic Anthropometric Prior (KAP) | Does KAP improve generalisation? | v23 |
| v18 + fixed BA + KAP | Does bundle-adjustment refinement help? | v24 |
| v18 + neural BA | Can a neural BA surrogate improve accuracy? | v21 |
| Bayesian Tri v2 (d=128) | Best geometry-first configuration | v2 anchor |

### 5.4 Robustness and calibration tests
- 2D keypoint Gaussian noise.
- Random joint occlusion.
- Random view dropout.
- Controlled camera perturbations: rotation, translation, focal length, principal point.
- Variable-view inference (k = 2..14).

### 5.5 Cross-dataset and transfer tests
- Train on MPI, test on H36M / 3DPW / AIST++ / WebBridge.
- Mixed-dataset training (MPI + H36M).

### 5.6 Runtime and integration
- Latency/throughput on RTX 4090 (batch 1–16).
- MotionFlow plugin demo with uncertainty and provenance logging.

---

## 6. Results to Highlight

### 6.1 Anchor numbers
| Dataset / Condition | Model | MPJPE (mm) | PA-MPJPE (mm) |
|---------------------|-------|-----------:|----------------:|
| MPI-INF-3DHP S2/Seq1 | Raw DLT | 25.21 | 24.08 |
| MPI-INF-3DHP S2/Seq1 | Cross-view residual + PP (d=64) | **9.32** | **5.37** |
| MPI-INF-3DHP S2/Seq1 | Bayesian Tri v2 ensemble (d=128) | **8.35** | **5.29** |
| Human3.6M S5/Act2 | Cross-view residual + PP (d=64) | **5.24** | **4.84** |

### 6.2 Calibration robustness snapshot
| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|-----------|-----------:|----------------:|
| clean | 9.32 | 5.37 |
| rot_0.5° | 16.89 | 8.11 |
| trans_5mm | 10.61 | 5.20 |
| focal_1% | 19.13 | 8.07 |
| cxcy_3px | 11.41 | 5.75 |

### 6.3 Extended robustness snapshot (Bayesian Tri v2 d=128)
| Condition | MPJPE (mm) | PCK@50 | AUC |
|-----------|-----------:|-------:|----:|
| clean | 9.03 | 1.000 | 0.940 |
| noise_2.0px | 9.31 | 1.000 | 0.938 |
| joint_occlusion_20 | 14.56 | 0.999 | 0.903 |
| view_dropout_30 | 18.15 | 0.995 | 0.879 |

---

## 7. Discussion

- **What works:** geometry-first decomposition, PP correction, compact model size, epipolar auxiliary loss.
- **What remains hard:** rotation/focal drift, variable-view inference with very few views, cross-dataset transfer.
- **Future work:** rotation correction head, adaptive view selection, skeleton-graph refiner, SMPL fitting stage, domain adaptation.

---

## 8. Conclusion

- Recap the *triangulate first, then residual* principle.
- State the practical contribution: a calibrated, uncertainty-aware, plug-in multi-view fusion module suitable for robotics and immersive video.
- Reiterate key results and call to next experiments (v23/v24, full-data runs, test-set evaluation).

---

## 9. Planned Experiments (as of 2026-08-08)

| Experiment | Description | Status | GPU | Paper section |
|------------|-------------|--------|-----|---------------|
| **v23** | v18 + KAP, no neural BA | running | A800 GPU4/GPU6 | §5.3 ablation |
| **v18 full** | Cross-view residual + PP baseline | running | A800 GPU5 | §5.2 main results |
| **v11 full** | Temporal ray-attention, no residual | running | A800 GPU7 | §5.3 ablation |
| **v21** | v18 + neural BA surrogate | stopped at 128.27 mm (regressed) | — | §5.3 negative result |
| **v24** | v18 + fixed BA + KAP | prepared, queued | — | §5.3 ablation |
| **Bayesian Tri v2 d=128** | Best geometry-first anchor | done (8.35 mm ensemble) | — | §5.2 / §5.4 |

**Next concrete step:** wait for v23 to complete its first-epoch validation MPJPE; if it improves on v18, use it to either replace the v18 anchor in the main-results table or add it as a positive ablation, then launch v24 on the next free A800 GPU.

---

## 10. Figures and Tables to Prepare

1. **Architecture diagram:** 2D keypoints → intrinsic correction → ray embedding → (T×V) attention → covariance/weight heads → weighted DLT → adaptive Gauss-Newton → residual MLP → 3D pose + uncertainty.
2. **Main results table:** MPI-INF-3DHP and Human3.6M MPJPE/PA-MPJPE for DLT, v11, v18, v23/v24, and Bayesian Tri v2.
3. **Calibration robustness heatmap:** MPJPE under rot/trans/focal/PP perturbations.
4. **Variable-view curve:** MPJPE@k for k = 2..14.
5. **Runtime plot:** latency/throughput on RTX 4090 for batch 1–16.
6. **Failure-case heatmaps:** per-joint/per-view error and PP correction magnitude.
