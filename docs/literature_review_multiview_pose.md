# Literature Review: Multi-View Human Pose Estimation (2023–2026)

> Scope: Recent CVPR / ICCV / ECCV / ICRA and top-tier arXiv works on multi-view 3D human pose estimation, with a focus on methods relevant to the MotionFlow-MultiView project (ICRA/CVPR 2027).
> Date: 2026-08-07
> Repo: `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`

---

## 1. Introduction

Multi-view 3D human pose estimation has moved well beyond the classical “detect-2D-then-triangulate” pipeline. Since 2023, the dominant trends are:

1. **Geometry-aware transformers** that inject camera parameters, epipolar constraints, or ray embeddings into attention.
2. **Uncertainty / occlusion-aware fusion** that learns per-view visibility and confidence.
3. **Temporal / state-space models** that exploit video continuity.
4. **Generative and self-supervised approaches** that reduce dependence on scarce 3D annotations.
5. **Calibration-robust methods** that tolerate imperfect or unknown camera parameters.

This review maps these trends, cites concrete papers, and explains how each relates to the MotionFlow-MultiView v5 stack.

---

## 2. Quick Taxonomy

| Theme | Representative Works | Key Idea | Relevance to MotionFlow-MV |
|---|---|---|---|
| Geometry-biased transformers | Moliner & Huang (FG 2024); Liao & Zhu (arXiv 2023) | Inject projective geometry into multi-view attention | Justifies our ray/camera embeddings and epipolar losses |
| Uncertainty & occlusion fusion | Bragagnolo et al. (ECCVW 2024); Ghasemzadeh & Alahi (ECCV T-CAP 2024); Davoodnia et al. (ECCV 2024) | Learn per-view weights or visibility gates | Directly aligned with v2 visibility gating and Bayesian Tri v2 covariance weights |
| Temporal / state-space | Choudhury & Kitani (ICCV 2023); Chharia et al. (CVPR 2025) | Recurrent/state-space fusion across time | Supports our v13 temporal consistency and v19 perceiver directions |
| Ray-based lifting | Ghasemzadeh & Alahi (arXiv 2025) | Ray tokens + transformers for 2D-to-3D lifting | Informs v17 cross-view transformer design |
| Generative & diffusion | Wang & Birdal (arXiv 2026); Bragagnolo et al. (WACV 2026) | Diffusion/Gaussian splatting for pose refinement | Potential v20 diffusion refiner and v21 bundle-adjustment priors |
| Self-supervised / weakly supervised | Li & Meng (arXiv 2024); Srivastav et al. (arXiv 2024); Wang & Birdal (arXiv 2026) | Canonical spaces, multi-view consistency, hypergraph cover | Guides data-efficiency and WebBridge analysis |
| Calibration | Jiang & Hu (arXiv 2023); Tang et al. (arXiv 2024); Tuo (arXiv 2026) | Uncalibrated / sparse calibration, RANSAC-P3P | Links to our v10–v12 calibration-robust baselines |

---

## 3. Geometry-Biased Transformers & Learned Triangulation

### Moliner & Huang, “Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction,” FG 2024

- **Problem:** Occlusion and limited overlapping views.
- **Approach:** An encoder-decoder Transformer whose encoder refines 2D skeleton joints across views, then decodes to 3D.
- **Relation to MotionFlow-MV:** Reinforces the value of skeleton-aware cross-view attention, similar to our GraphJointAttentionV2 and v3 hierarchical fusion. Their geometry bias motivates our epipolar-biased transformer layer in v3.

### Liao & Zhu, “Multiple View Geometry Transformers for 3D Human Pose Estimation,” arXiv 2023

- **Problem:** End-to-end transformers struggle with projective geometry under occlusion.
- **Approach:** MVGFormer interleaves geometric and appearance modules.
- **Relation to MotionFlow-MV:** Supports our hybrid triangulation-transformer design (Bayesian Tri v2 + OmniMultiViewFusion v5). Pure regression is not enough; geometry modules are essential.

### Jiang & Hu, “Probabilistic Triangulation for Uncalibrated Multi-View 3D Human Pose Estimation,” arXiv 2023

- **Problem:** Standard multi-view pipelines require fixed, calibrated cameras.
- **Approach:** A Probabilistic Triangulation module that can be embedded in a calibration-free network.
- **Relation to MotionFlow-MV:** Closely aligned with our variable-view and calibration-robust goals. Our v10–v12 baselines already explore camera-perturbation curricula; this paper provides a theoretical grounding for uncalibrated triangulation.

---

## 4. Occlusion-Aware & Uncertainty-Aware Fusion

### Bragagnolo, Terreran et al., “Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation,” ECCV Workshops 2024

- **Problem:** Strong occlusions and limited camera viewpoints in human-robot collaboration.
- **Approach:** Fuses multi-view 2D features instead of noisy triangulation, explicitly reasoning about occlusion.
- **Relation to MotionFlow-MV:** Directly comparable to our OmniMultiViewFusion v2 visibility gating. The paper’s human-robot focus also matches ICRA relevance.

### Ghasemzadeh & Alahi, “MPL: Lifting 3D Human Pose from Multi-view 2D Poses,” ECCV T-CAP 2024

- **Problem:** Lack of in-the-wild multi-view 3D training data.
- **Approach:** Combines off-the-shelf 2D pose estimation with a learned 2D-to-3D lifter trained on synthetic data.
- **Relation to MotionFlow-MV:** Similar to our 2D-keypoint → 3D-pose pipeline. Their use of synthetic training data is relevant to our WebBridge and H36M mixed-data strategy.

### Davoodnia, Ghorbani et al., “UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues,” ECCV 2024

- **Problem:** Need robustness without direct 3D annotations.
- **Approach:** A pose compiler refines single-image 2D keypoints using temporal and cross-view cues; introduces a scalable cross-view fusion strategy.
- **Relation to MotionFlow-MV:** Mirrors our cross-view + temporal loss design. Their uncertainty weighting is a simpler version of our anisotropic covariance in Bayesian Tri v2.

### Jiao, Cheng, Yang et al., “DeProPose: Deficiency-Proof 3D Human Pose Estimation via Adaptive Multi-View Fusion,” arXiv 2025

- **Problem:** Occlusion, noise, missing viewpoints.
- **Approach:** Deficiency-aware adaptive multi-view fusion.
- **Relation to MotionFlow-MV:** Their “deficiency-proof” goal is exactly our robustness matrix motivation (variable views, occlusion, camera perturbation).

---

## 5. Temporal, State-Space & Long-Clip Models

### Choudhury & Kitani, “TEMPO: Efficient Multi-View Pose Estimation, Tracking, and Forecasting,” ICCV 2023

- **Problem:** Volumetric methods are accurate but expensive and single time-step only.
- **Approach:** Recurrent 2D pose features fused across space and time, also producing tracking and forecasting.
- **Relation to MotionFlow-MV:** Temporal consistency is a core v13 baseline (velocity + acceleration losses). TEMPO justifies expanding our temporal refiner into tracking/forecasting extensions.

### Chharia, Gou, Dong et al., “MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation,” CVPR 2025

- **Problem:** Attention-based transformers overfit specific camera arrangements and struggle with occlusions.
- **Approach:** Replace attention with multi-view state-space modeling (SSM) for better spatial arrangement and cross-view generalization.
- **Relation to MotionFlow-MV:** Suggests an alternative to our transformer-based fusion. Could inform a future v23 “Mamba-style” cross-view aggregator if attention proves too view-count sensitive.

### Li, Chen, Wang et al., “From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper,” arXiv 2026

- **Problem:** Standard methods use simultaneous multi-view frames, ignoring temporal dependencies.
- **Approach:** Sparse interleaved input and DenseWarper module that warps dense temporal features across views.
- **Relation to MotionFlow-MV:** Supports our v13/v16 temporal augmentation and points toward denser temporal fusion than our current velocity/acceleration losses.

---

## 6. Ray-Based Lifting & Egocentric Multi-View

### Ghasemzadeh & Alahi, “RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting,” arXiv 2025

- **Problem:** Generalization to unseen real-world multi-view rigs.
- **Approach:** Ray-based tokens and a transformer lifter; decouples 2D detection from 3D lifting.
- **Relation to MotionFlow-MV:** Strong justification for our v17 cross-view transformer operating on ray/camera embeddings. Their “universal” goal matches our variable-view-count design.

### Yang, Tkach et al., “EgoPoseFormer: A Simple Baseline for Stereo Egocentric 3D Human Pose Estimation,” ECCV 2024

- **Problem:** Self-occlusion and limited FOV in egocentric stereo cameras.
- **Approach:** Two-stage transformer: coarse global pose then refinement.
- **Relation to MotionFlow-MV:** Their handling of invisible joints via global context is analogous to our graph-joint attention and principal-point correction.

### Akada, Wang, Golyanik, “Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation,” arXiv 2025

- **Problem:** Frontal HMD cameras alone are insufficient for full-body tracking.
- **Approach:** Exploits additional rear body-worn cameras.
- **Relation to MotionFlow-MV:** Highlights the importance of camera-layout diversity; our variable-view training can simulate such non-frontal setups.

---

## 7. Generative, Diffusion & Differentiable Rendering

### Wang, Birdal, Navab, “DisPOSE: Projected Polystochastic Diffusion for Self-Supervised Multi-View 3D Human Pose Estimation,” arXiv 2026

- **Problem:** Self-supervised multi-view person association and 3D pose for interacting people.
- **Approach:** Diffusion process over projected person assignments (polystochastic).
- **Relation to MotionFlow-MV:** Relevant if we extend from single-person to multi-person multi-view. Provides a generative prior that could augment our v20 diffusion refiner.

### Bragagnolo, Barcellona, Ghidoni et al., “SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering,” WACV 2026

- **Problem:** Learned methods overfit training scenarios and generalize poorly.
- **Approach:** 3D Gaussian splatting of a skeleton, optimized via differentiable rendering.
- **Relation to MotionFlow-MV:** A rendering-based prior could regularize our neural bundle adjustment (v21) or SMPL-prior fusion (v22).

### Heo, Wang, Liu et al., “Motion Diffusion-Guided 3D Global HMR from a Dynamic Camera,” arXiv 2024

- **Problem:** Monocular global human mesh recovery from dynamic cameras.
- **Approach:** Diffusion model guided by motion cues.
- **Relation to MotionFlow-MV:** Although monocular, the diffusion-guided refinement idea directly maps to our v20 diffusion pose refiner for post-triangulation polishing.

---

## 8. Self-Supervised, Weakly Supervised & Data-Efficient Methods

### Li & Meng, “Self-learning Canonical Space for Multi-view 3D Human Pose Estimation,” arXiv 2024

- **Problem:** Accurate multi-view annotations are scarce.
- **Approach:** Learns a canonical pose space from multi-view consistency without full 3D labels.
- **Relation to MotionFlow-MV:** Complementary to our WebBridge + H36M mixed training; a canonical-space loss could improve cross-dataset generalization.

### Srivastav et al., “SelfPose3d: Self-Supervised Multi-Person Multi-View 3d Pose Estimation,” arXiv 2024

- **Problem:** Scaling multi-person multi-view 3D pose without 3D annotations.
- **Approach:** Self-supervised consistency across views.
- **Relation to MotionFlow-MV:** Provides a roadmap for extending our single-person pipeline to multi-person scenes without 3D labels.

### Wang, Birdal, Navab, “COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation,” arXiv 2026

- **Problem:** Sparse multi-view rigs and lack of 3D supervision.
- **Approach:** Hypergraph cover optimization to solve correspondence and pose jointly.
- **Relation to MotionFlow-MV:** Their training-free optimization angle informs our neural bundle adjustment (v21) and robust triangulation baselines.

---

## 9. Calibration & Camera Robustness

### Tang, Suri et al., “CasCalib: Cascaded Calibration for Motion Capture from Sparse Unsynchronized Cameras,” arXiv 2024

- **Problem:** Calibration of sparse, unsynchronized camera rigs.
- **Approach:** Cascaded calibration pipeline tailored for motion capture.
- **Relation to MotionFlow-MV:** Our v10–v12 experiments add camera perturbations. CasCalib-style recalibration could be integrated as a preprocessing stage or a learned refinement head.

### Tuo, “RPGD: RANSAC-P3P Gradient Descent for Extrinsic Calibration in 3D Human Pose Estimation,” arXiv 2026

- **Problem:** Extrinsic calibration errors degrade pose accuracy.
- **Approach:** Hybrid RANSAC + P3P gradient descent.
- **Relation to MotionFlow-MV:** Reinforces the value of camera-parameter conditioning and principal-point correction in our v3/v5 architecture.

---

## 10. Relation to MotionFlow-MultiView v5

| Project Component | How the Literature Supports / Challenges It |
|---|---|
| **Bayesian Tri v2** (anisotropic covariance, weighted DLT, adaptive Gauss-Newton) | Builds on Isakov et al. (ICCV 2019) and Jiang & Hu (arXiv 2023); UPose3D and DeProPose show that uncertainty-aware fusion is now a mainstream requirement. |
| **OmniMultiViewFusion v2 visibility gating** | Direct counterpart to Bragagnolo et al. (ECCVW 2024) and MPL. Our per-joint visibility multiplier is more explicit and can be combined with their feature-level fusion. |
| **Graph-joint / skeleton attention** | Geometry-Biased Transformer and MVGFormer validate skeleton-aware attention. Our GraphJointAttentionV2 is a concrete instance. |
| **Epipolar-biased transformer (v3)** | Geometry-biased and ray-based papers (MVGFormer, RUMPL) provide theoretical support for injecting epipolar/ray embeddings into attention. |
| **Temporal consistency (v13)** | TEMPO and MV-SSM show temporal/state-space models improve multi-view pose; our v13 uses simpler velocity/acceleration losses, leaving room for a full perceiver or SSM. |
| **Variable views / calibration robustness** | Probabilistic Triangulation, CasCalib, RPGD, and MV-SSM all target this. It is the strongest differentiator of our v10–v16 baselines. |
| **Diffusion / bundle adjustment (v20–v21)** | DisPOSE, SkelSplat, and Motion Diffusion-Guided HMR suggest generative/rendering priors can refine triangulated poses. |

---

## 11. Open Opportunities for ICRA/CVPR 2027

1. **Tighter geometry-attention coupling:** Move from separate triangulation + attention blocks to a single transformer that attends over rays and jointly refines cameras + pose (v17 / v21).
2. **State-space cross-view fusion:** Evaluate MV-SSM-style SSMs as a drop-in replacement for attention when view count varies widely.
3. **Diffusion refinement:** Add a lightweight diffusion head after Bayesian Tri v2 for outlier suppression and plausible pose completion (v20).
4. **Self-supervised pretraining on WebBridge:** Use canonical-space or hypergraph-cover losses to leverage unlabeled or weakly labeled multi-view video (Li & Meng; COMPOSE).
5. **Multi-person extension:** DisPOSE and SelfPose3d provide a path from our single-person system to interacting subjects.

---

## References

1. Bragagnolo, L., Terreran, M., et al. *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation.* ECCV Workshops, 2024. arXiv:2408.15810
2. Ghasemzadeh, S. A., Alahi, A., et al. *MPL: Lifting 3D Human Pose from Multi-view 2D Poses.* ECCV T-CAP, 2024. arXiv:2408.10805
3. Davoodnia, V., Ghorbani, S., et al. *UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues.* ECCV, 2024. arXiv:2404.14634
4. Yang, C., Tkach, A., et al. *EgoPoseFormer: A Simple Baseline for Stereo Egocentric 3D Human Pose Estimation.* ECCV, 2024. arXiv:2403.18080
5. Li, X., Meng, M., et al. *Self-learning Canonical Space for Multi-view 3D Human Pose Estimation.* arXiv:2403.12440, 2024.
6. Srivastav, V., Chen, K., et al. *SelfPose3d: Self-Supervised Multi-Person Multi-View 3d Pose Estimation.* arXiv:2404.02041, 2024.
7. Tang, J., Suri, S., et al. *CasCalib: Cascaded Calibration for Motion Capture from Sparse Unsynchronized Cameras.* arXiv:2405.06845, 2024.
8. Heo, J., Wang, K.-C., Liu, K., et al. *Motion Diffusion-Guided 3D Global HMR from a Dynamic Camera.* arXiv:2411.10582, 2024.
9. Choudhury, R., Kitani, K., et al. *TEMPO: Efficient Multi-View Pose Estimation, Tracking, and Forecasting.* ICCV, 2023. arXiv:2309.07910
10. Moliner, O., Huang, S., et al. *Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction.* FG, 2024. arXiv:2312.17106
11. Liao, Z., Zhu, J., et al. *Multiple View Geometry Transformers for 3D Human Pose Estimation.* arXiv:2311.10983, 2023.
12. Jiang, B., Hu, L., et al. *Probabilistic Triangulation for Uncalibrated Multi-View 3D Human Pose Estimation.* arXiv:2309.04756, 2023.
13. Xu, Y., Kitani, K., et al. *Multi-View Person Matching and 3D Pose Estimation with Arbitrary Uncalibrated Camera Networks.* arXiv:2312.01561, 2023.
14. Chharia, A., Gou, W., Dong, H., et al. *MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation.* CVPR, 2025. arXiv:2509.00649
15. Ghasemzadeh, S. A., Alahi, A., et al. *RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting.* arXiv:2512.15488, 2025.
16. Jiao, J., Cheng, X., Yang, K., et al. *DeProPose: Deficiency-Proof 3D Human Pose Estimation via Adaptive Multi-View Fusion.* arXiv:2502.16419, 2025.
17. Akada, H., Wang, J., Golyanik, V., et al. *Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation.* arXiv:2503.11652, 2025.
18. Bragagnolo, L., Barcellona, L., Ghidoni, S., et al. *SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering.* WACV, 2026. arXiv:2511.08294
19. Wang, T. D., Birdal, T., Navab, N., et al. *COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation.* arXiv:2601.09698, 2026.
20. Wang, T. D., Birdal, T., Navab, N., et al. *DisPOSE: Projected Polystochastic Diffusion for Self-Supervised Multi-View 3D Human Pose Estimation.* arXiv:2606.07419, 2026.
21. Li, L., Chen, C., Wang, Y., et al. *From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper.* arXiv:2605.14525, 2026.
22. Tuo, Z. *RPGD: RANSAC-P3P Gradient Descent for Extrinsic Calibration in 3D Human Pose Estimation.* arXiv:2602.13901, 2026.

---

*This file was produced by the `docs_literature_review` swarm task for ICRA/CVPR 2027 planning.*
