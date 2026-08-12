# Related-Work Survey: Recent CVPR / ICRA / RA-L Multi-View Human Pose Methods

> Scope: CVPR 2024–2025, ICRA / RA-L 2025, and closely adjacent top-tier venues (ECCV 2024, WACV 2026) for multi-view 3D human pose estimation.
> Date: 2026-08-08
> Repo: `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`

This survey feeds the ICRA / CVPR 2027 story of MotionFlow-MultiView. We focus on methods that are (a) recent, (b) multi-view, and (c) most relevant to our geometry-first, calibration-robust, uncertainty-aware pipeline.

---

## 1. Taxonomy of Recent Methods

| Theme | Representative Works | Key Idea | MotionFlow-MV Take-away |
|---|---|---|---|
| **State-space / sequence models** | MV-SSM (CVPR 2025) | Replace vanilla cross-view attention with state-space blocks | Alternative to transformers for variable-view fusion |
| **Self-supervised multi-person pose** | SelfPose3D (CVPR 2024) | Multi-view consistency losses without 3D labels | Data-efficiency and WebBridge relevance |
| **Occlusion-aware skeleton fusion** | Bragagnolo et al. (ECCVW 2024) | Fuse 3D skeletons (not 2D heatmaps) under occlusion | Reinforces our cross-view residual + visibility gating |
| **Uncertainty + temporal fusion** | UPose3D (ECCV 2024) | Cross-view + temporal cues with uncertainty weights | Precedent for our Bayesian Tri + temporal consistency |
| **Ray-based universal lifting** | RUMPL (arXiv 2025) | Ray tokens + transformer, view-count agnostic | Strong justification for our ray/camera embeddings |
| **Deficiency-aware fusion** | DeProPose (arXiv 2025) | Relative-projection-error weights for noisy/missing views | Directly maps to our robustness matrix experiments |
| **Training-free optimization** | COMPOSE (arXiv 2026) | Hypergraph exact-cover over person hypotheses | Relevant for neural BA / robust triangulation fallback |
| **Generative assignment** | DisPOSE (arXiv 2026) | Diffusion over multi-view person assignments | Multi-person extension, generative priors |
| **Differentiable rendering** | SkelSplat (WACV 2026) | 3D Gaussian skeleton + differentiable rendering | Prior for neural BA / SMPL fusion |
| **Calibration from people** | Spatiotemporal Multi-Camera Calibration (RA-L 2025) | Joint calibration + association from freely moving people | Supports our calibration-robust / auto-calib agenda |

---

## 2. CVPR 2024–2025

### 2.1 MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation (CVPR 2025)

- **Authors:** Aviral Chharia, Wenbo Gou, Haoye Dong, et al.
- **Link:** https://arxiv.org/abs/2509.00649
- **Problem:** Standard cross-view transformers overfit to specific camera arrangements and struggle with occlusion.
- **Method:**
  - Introduces a **Projective State Space (PSS)** block that models joint spatial sequences at two levels: multi-view feature level and person keypoint level.
  - Adapts Mamba scanning into a **Grid Token-guided Bidirectional Scan (GTBS)** to preserve spatial structure.
  - State-space modeling replaces full self-attention for better scaling and generalization.
- **Results:** +10.8 AP25 on CMU Panoptic three-camera setting; strong cross-dataset transfer.
- **MotionFlow-MV Relevance:**
  - Validates that **attention is not the only cross-view aggregator** — state-space blocks are a viable v23/v24 alternative.
  - Their variable-view-count generalization supports our variable-view training.

### 2.2 SelfPose3d: Self-Supervised Multi-Person Multi-View 3D Pose Estimation (CVPR 2024)

- **Authors:** Srivastav et al.
- **Link:** https://arxiv.org/abs/2404.02041
- **Problem:** Scaling multi-person multi-view 3D pose without dense 3D annotations.
- **Method:**
  - Uses multi-view photometric / geometric consistency as supervision.
  - Learns 3D pose from 2D detections across views without ground-truth 3D labels.
- **Results:** Competitive with supervised methods on standard multi-person benchmarks.
- **MotionFlow-MV Relevance:**
  - Provides a roadmap for extending our single-person pipeline to multi-person scenes with limited 3D labels.
  - Self-supervised consistency could augment our WebBridge + H36M mixed-data strategy.

---

## 3. ICRA / RA-L 2025

### 3.1 Spatiotemporal Multi-Camera Calibration using Freely Moving People (RA-L 2025)

- **Authors:** Chen et al.
- **Link:** https://arxiv.org/abs/2502.12546
- **Problem:** Calibrating multiple cameras with unknown time offsets and associations using only freely moving people.
- **Method:**
  - Treats calibration + temporal alignment + person association as a single 3D point-registration problem.
  - Monocular 3D pose estimates are projected onto a unit sphere.
  - Alternates rotation/time-offset/association solving with soft assignment, then refines with global non-linear optimization.
- **Results:** Marker-free calibration competitive with classical target-based methods.
- **MotionFlow-MV Relevance:**
  - Aligns directly with our **v10–v12 calibration-robust** and **v21/v24 bundle-adjustment** directions.
  - Suggests a practical pre-processing step for uncalibrated / drifting multi-camera rigs.

### 3.2 ICRA 2025 Adjacent Works (Monocular / Stereo Pose)

- **Toward a Real-Time Framework for Accurate Monocular 3D Human Pose Estimation with Geometric Priors** (ICRA 2025 workshop)
  - Uses geometric priors to constrain monocular 3D pose.
  - Relevant to us as a **fallback when only one view is reliable** in our variable-view inference.
- **Systematic Comparison of Projection Methods for Monocular 3D Human Pose Estimation on Fisheye Images** (ICRA 2025)
  - Compares perspective, equiangular, and Kannala-Brandt projection models for fisheye pose.
  - Relevant to our camera-model conditioning when wide-angle or fisheye cameras are present.

---

## 4. Adjacent Top-Tier and Strong arXiv Works

### 4.1 Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation (ECCVW 2024)

- **Authors:** Bragagnolo, Terreran, Allegro, Ghidoni
- **Link:** https://arxiv.org/abs/2408.15810
- **Method:** Fuses multi-view 3D skeletons (output by monocular absolute pose estimators) instead of noisy 2D features, and optimizes with limb-length symmetry constraints.
- **MotionFlow-MV Relevance:**
  - Directly comparable to our **OmniMultiViewFusion v2 visibility gating** and **v18 cross-view residual + PP correction**.
  - Reinforces that **occlusion handling** is a key differentiator.

### 4.2 UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues (ECCV 2024)

- **Authors:** Davoodnia, Ghorbani, et al.
- **Link:** https://arxiv.org/abs/2404.14634
- **Method:** Refines 2D keypoints using both cross-view and temporal cues, with explicit uncertainty weighting.
- **MotionFlow-MV Relevance:**
  - Mirrors our **Bayesian Tri v2** and **v13 temporal consistency** directions.
  - Their uncertainty weighting is a simpler precursor to our anisotropic covariance formulation.

### 4.3 RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting (arXiv 2025)

- **Authors:** Ghasemzadeh, Alahi, De Vleeschouwer
- **Link:** https://arxiv.org/abs/2512.15488
- **Method:** Represents 2D keypoints as 3D rays; view-fusion transformer aggregates along rays; camera- and view-count agnostic.
- **MotionFlow-MV Relevance:**
  - Strong justification for our **v17 ray/camera embeddings** and variable-view design.
  - Achieves up to 53% MPJPE reduction vs. triangulation.

### 4.4 DeProPose: Deficiency-Proof 3D Human Pose Estimation via Adaptive Multi-View Fusion (arXiv 2025)

- **Authors:** Jiao, Cheng, Yang, et al.
- **Link:** https://arxiv.org/abs/2502.16419
- **Method:** End-to-end multi-view fusion using **relative projection error** to dynamically weight views; introduces DA-3DPE dataset for noise, missing views, and occlusion.
- **MotionFlow-MV Relevance:**
  - Directly maps to our **extended robustness matrix** (noise, joint occlusion, view dropout).
  - Supports the need for adaptive view weighting in v18/v23.

### 4.5 COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation (arXiv 2026)

- **Authors:** Wang, Birdal, Navab, Bastian
- **Link:** https://arxiv.org/abs/2601.09698
- **Method:** Training-free hypergraph exact-cover optimization over person hypotheses; replaces pairwise association with a global combinatorial objective solved by ILP / Belief Propagation.
- **MotionFlow-MV Relevance:**
  - Informs our **neural BA / robust triangulation** fallback strategies.
  - Could be a strong baseline for multi-person extension without 3D labels.

### 4.6 DisPOSE: Projected Polystochastic Diffusion for Self-Supervised Multi-View 3D Human Pose Estimation (arXiv 2026)

- **Authors:** Wang, Birdal, Navab, Bastian
- **Link:** https://arxiv.org/abs/2606.07419
- **Method:** Diffusion over polystochastic tensors for multi-view person assignment; hypergraph-convolutional decoder regresses 3D skeletons.
- **MotionFlow-MV Relevance:**
  - Generative assignment prior could extend our pipeline to **multi-person scenes**.
  - Diffusion-based refinement aligns with our **v20 diffusion refiner** line of work.

### 4.7 SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering (WACV 2026)

- **Authors:** Bragagnolo, Barcellona, Ghidoni, et al.
- **Link:** https://arxiv.org/abs/2511.08294
- **Method:** Models skeleton as 3D Gaussians and optimizes via differentiable rendering; no 3D ground truth required.
- **MotionFlow-MV Relevance:**
  - Differentiable rendering prior could regularize our **v21 neural BA** or **v22/v23 SMPL / KAP** branches.
  - Cross-dataset robustness is a goal we share.

---

## 5. Synthesis: What This Means for MotionFlow-MV v23/v24

| Our Component | Supported By | Challenge / Gap |
|---|---|---|
| **Bayesian Tri v2** (anisotropic covariance, weighted DLT) | UPose3D, DeProPose | Need to validate on RA-L / CVPR-style calibration-robust benchmarks |
| **Ray/camera embeddings** (v17) | RUMPL | RUMPL shows universal lifting is possible; we should test variable-view generalization |
| **Cross-view residual + PP correction** (v18) | Bragagnolo et al. ECCVW 2024 | Their skeleton fusion is a strong external baseline to cite |
| **Temporal consistency** (v13) | UPose3D, MV-SSM | MV-SSM suggests a state-space alternative to our transformer temporal head |
| **Kinematic Anthropometric Prior** (v22/v23) | SkelSplat, COMPOSE | Bone-length / skeleton priors are now mainstream; KAP must show clear lift over v18 |
| **Calibration robustness / fixed BA** (v24) | RA-L calibration paper, COMPOSE | Auto-calibration from people is a practical future extension |
| **Multi-person extension** | SelfPose3D, DisPOSE, COMPOSE | Not yet in our pipeline; strongest longer-term direction |

---

## 6. Open Questions for ICRA / CVPR 2027

1. **Can a state-space cross-view module (a la MV-SSM) outperform our transformer fusion on variable-view inference?**
2. **Should we adopt ray-based universal lifting (RUMPL) to make the pipeline camera- and view-count agnostic?**
3. **How do we incorporate training-free optimization (COMPOSE) as a fallback when learned confidence is low?**
4. **Can we extend the single-person pipeline to multi-person using SelfPose3D-style consistency or DisPOSE-style generative assignment?**
5. **What calibration-robust experiments are needed to compete with the RA-L 2025 auto-calibration work?**

---

## 7. References (Selected)

- Chharia et al., "MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation," CVPR 2025.
- Srivastav et al., "SelfPose3d: Self-Supervised Multi-Person Multi-View 3d Pose Estimation," CVPR 2024.
- Bragagnolo et al., "Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation," ECCVW 2024.
- Davoodnia et al., "UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues," ECCV 2024.
- Ghasemzadeh & Alahi, "RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting," arXiv 2025.
- Jiao et al., "DeProPose: Deficiency-Proof 3D Human Pose Estimation via Adaptive Multi-View Fusion," arXiv 2025.
- Wang & Birdal et al., "COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation," arXiv 2026.
- Wang & Birdal et al., "DisPOSE: Projected Polystochastic Diffusion for Self-Supervised Multi-View 3D Human Pose Estimation," arXiv 2026.
- Bragagnolo et al., "SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering," WACV 2026.
- Chen et al., "Spatiotemporal Multi-Camera Calibration using Freely Moving People," IEEE RA-L 2025.
