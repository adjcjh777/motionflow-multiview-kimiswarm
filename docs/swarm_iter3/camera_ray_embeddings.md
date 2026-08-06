# Camera-Ray Embeddings for Multi-View Human Pose Estimation

> Research topic for the motionflow-multiview fusion pipeline.  
> Target venues: CVPR / ICRA 2027.  
> Prepared from the local codebase and general domain knowledge (no external web fetch available during this session).

## 1. Problem statement

In calibrated multi-view human pose estimation, a 2-D keypoint `(u, v)` together with a camera defines a **ray in 3-D world space**.  All geometrically consistent triangulations are equivalent to intersecting these rays.  However, the current learned fusion path in `motionflow_mv` lifts only the raw image coordinate `(x, y, confidence)` into a latent space; camera parameters enter either not at all (the original `attention` plugin) or as a flattened 12-vector projection matrix (`attention_v2`).  Neither representation makes the network explicitly reason about rays.

The **camera-ray embedding** problem is therefore: how should a camera and a 2-D observation be encoded so that a learned multi-view fusion module

1. respects projective / Euclidean geometry,
2. is invariant (or equivariant) to irrelevant changes such as image scale or global scene units, and
3. can generalize across datasets with different calibrations and focal lengths.

A good ray embedding should turn multi-view fusion into an attention-over-views problem in 3-D geometric space, rather than an attention-over-views problem in raw 2-D pixel space.

## 2. Key related work and methods

Below are the works most relevant to constructing and using camera-ray embeddings for multi-view 3-D human pose and SMPL fitting.

### 2.1 Learnable triangulation and volumetric methods

**Iskakov et al., "Learnable Triangulation of Human Pose," ICCV 2019.**  
This is the seminal work on replacing algebraic triangulation with a learned network.  It projects 2-D keypoints into 3-D voxel volumes using camera parameters, effectively encoding ray information into a volumetric representation.  The key insight — that 2-D detections should be lifted into 3-D according to calibrated rays — underlies modern geometry-aware fusion.

**Remelli et al., "Lightweight Multi-Camera 3D Human Pose Estimation," FG 2020.**  
Introduces a lightweight architecture that uses camera parameters to unproject 2-D heatmaps into a canonical 3-D volume.  The unprojection is exactly a ray-sampling operation: each voxel accumulates evidence along its corresponding camera ray.

### 2.2 Geometry-biased transformers

**Moliner et al., "Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction," arXiv 2023.**  
Already cited in `docs/swarm_iter1/08_attention_fusion.md` and closely aligned with our `attention_v2` direction.  The paper biases self-attention with geometric terms derived from camera rays, improving robustness to occlusion and few-view settings.  Their positional bias can be viewed as a form of ray embedding injected into the attention scores rather than into the token features.

**MVGFormer (Multi-View Geometry Transformer), ICCV 2023.**  
Builds a transformer that reasons directly over multi-view geometric features.  Camera rays are used both to initialize tokens and to enforce cross-view geometric consistency through epipolar constraints.  This is the closest architectural template for turning motionflow's per-joint attention into a geometry-aware transformer.

### 2.3 Ray-based neural fields and distance fields

**Sárközi et al., "PAnDA: Parametric 3D Distance Fields for 3D Human Pose Estimation," CVPR 2023.**  
Represents the human body as a 3-D distance field queried by camera rays.  Although primarily a single-view method, the idea of parameterizing queries by **ray origin + ray direction** is directly transferable to multi-view fusion: each view contributes a set of rays, and the network predicts the 3-D point that best explains them.

### 2.4 SMPL fitting with camera rays

**SMPLify / ProHMR / CLIFF.**  In SMPL fitting, reprojection loss is the standard way to enforce multi-view consistency.  The reprojection residual is the image-space distance between a 2-D detection and the projection of the 3-D model.  Recent work (CLIFF, HMR 2.0) shows that **camera-parameter embeddings** help networks disambiguate scale and perspective.  Extending this to multi-view means embedding the per-view ray associated with each detected joint and using it to constrain the SMPL body model.

## 3. Relation to the current motionflow-multiview codebase

### 3.1 Where ray embeddings fit

The relevant code paths are:

- `motionflow_mv/fusion/attention_model_v2.py` — the current geometry-aware attempt.
- `motionflow_mv/fusion/attention_fusion_v2_module.py` — plugin wrapper that flattens `P = K[R|t]` into a 12-vector.
- `motionflow_mv/fusion/robust_triangulation.py` — a learned per-view weighting over a differentiable DLT, already using `proj_matrices` correctly.
- `motionflow_mv/calibration/camera.py` — `Camera` exposes `projection_matrix`; it can also provide `K`, `R`, and `t`.

The current `AttentionFusionModelV2` (lines 22–39) does:
