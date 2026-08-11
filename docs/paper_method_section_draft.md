# Method

This section describes the multi-view human motion capture method at the core of MotionFlow-MultiView. We present the system architecture, formulate the pose estimation problem, and then detail each stage of the fusion module: intrinsic self-correction, visibility gating, skeleton-aware graph-joint attention, Bayesian precision-weighted triangulation, adaptive Gauss-Newton refinement, and residual refinement. We then describe training, inference, and the key differences from prior work.

\section{System Overview}

MotionFlow-MultiView is an end-to-end, reproducible multi-view human motion capture workflow. Its purpose is to turn calibrated or weakly calibrated multi-camera video into a robot-ready motion representation. The architecture is intentionally modular:

1. **Upstream: per-view frozen GVHMR.** A single-view human motion recovery model (GVHMR) runs independently on each camera view with frozen weights. This isolates the system contribution from upstream model improvements and provides a fair comparison of fusion modules.

2. **Fusion module: OmniMultiViewFusionV2.** The core contribution is a pluggable fusion module, implemented in `motionflow_mv/fusion/omniview_fusion_v2.py`, that fuses per-view 2D keypoints and confidences into a single metric 3D pose. It extends the Bayesian Tri v2 model in `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`.

3. **Intermediate representation: HumanMotionIR.** The fusion output is a stable, versioned schema called `HumanMotionIR` containing pose, uncertainty, quality flags, and provenance hashes.

4. **Downstream: robot profiles.** A robot-profile resolver externalizes degrees of freedom, joint names, action order, and export conventions so the same `HumanMotionIR` can drive different robots.

This design follows the principle that the fusion module should be geometry-first and learning-second: triangulate first, then learn only the structured residual error that geometry cannot fix.

\subsection{Honest true-GT framing}

The method is evaluated on non-circular ground truth: the 3D labels are independent mocap coordinates, not triangulations of the input 2D keypoints. On Human3.6M this means true-GT labels from `data/h36m_true_gt/`, not the circular `data/h36m_hf/` or `data/webbridge/h36m_meters/` sets. The method section therefore describes the architecture in terms of how it improves on geometric triangulation; the absolute MPJPE results in Section~5 show that, on honest labels, even our best learned variant still trails a simple confidence-weighted DLT baseline. The empirical contribution of the paper is accordingly framed around sparse-view and cross-domain robustness, not absolute records.

\section{Detailed Method}

\subsection{Notation and Problem Formulation}

A capture session consists of $V$ calibrated pinhole cameras observing a single person over $T$ frames. Let

- $\mathbf{K}_v \in \mathbb{R}^{3 \times 3}$ be the intrinsic matrix of view $v$,
- $\mathbf{R}_v \in SO(3)$ and $\mathbf{t}_v \in \mathbb{R}^3$ be the extrinsic rotation and translation,
- $\mathbf{x}_{t,v,j} = (u, v, c) \in \mathbb{R}^3$ be the detected 2D keypoint and detector confidence for joint $j$ in view $v$ at time $t$,
- $\mathbf{X}_{t,j} \in \mathbb{R}^3$ be the unknown world-coordinate 3D joint.

The input tensor has shape $(B, T, V, J, 3)$, where $B$ is the batch size and $J$ is the number of joints. The goal is to estimate the 3D skeleton

$$
\mathbf{X} = \{\mathbf{X}_{t,j}\}_{t=1,j=1}^{T,J} \in \mathbb{R}^{B \times T \times J \times 3}.
$$

We denote the projection matrix $\mathbf{P}_v = \mathbf{K}_v [\mathbf{R}_v \mid \mathbf{t}_v] \in \mathbb{R}^{3 \times 4}$.

\subsection{Input Representation}

The fusion module consumes two inputs:

1. **2D keypoints and confidences:** a tensor of shape $(B, T, V, J, 3)$ where the last channel stores $(u, v, c)$. Low-confidence detections ($c < 10^{-6}$) are masked out.

2. **Camera parameters:** either a list of `Camera` objects or batched tensors $(\mathbf{K}, \mathbf{R}, \mathbf{t})$ with shapes $(B, V, 3, 3)$, $(B, V, 3, 3)$, and $(B, V, 3)$ respectively. The module handles both single-rig and per-sample rigs.

Inside the model, each 2D point is back-projected to a unit ray using the (corrected) intrinsics, and the ray direction and camera center are embedded into a $d$-dimensional feature. This ray-aware representation is implemented in `omniview_fusion_v2.py` via `_extract_frame_features` and in the Bayesian Tri v2 model.

\subsection{Intrinsic Self-Correction}

Calibration drift is the dominant real-world failure mode of classical triangulation. The module therefore begins with a lightweight intrinsic correction head, `PrincipalPointCorrection`, that predicts per-view corrections to the principal point and, optionally, the focal length:

$$
\mathbf{K}_v^{\text{corr}} = \mathbf{K}_v + \Delta \mathbf{K}_v, \quad \Delta \mathbf{K}_v = f_{\text{pp}}(\{u_{v,j}, v_{v,j}, c_{v,j}\}_{j=1}^{J}).
$$

The head is a small MLP with hidden dimension `principal_point_hidden=64` that outputs a bounded offset $(\Delta c_x, \Delta c_y)$ and, if `focal_max_scale > 0`, a focal scale. The correction is supervised during training with the inverse of the applied perturbation, making the model self-calibrating at inference time. The corrected intrinsics are used for both ray embedding and triangulation. See `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`, lines 271--282.

\subsection{Visibility Gating for Occlusion Handling}

A dedicated `VisibilityGateHead` predicts a per-view, per-joint soft visibility multiplier $m_{t,v,j} \in [0,1]$. The head is context-aware: each view's visibility estimate is conditioned on the per-joint mean-pooled feature across all views, so the model exploits multi-view consistency when deciding whether a joint is occluded in a given view.

Let $\mathbf{f}_{t,v,j} \in \mathbb{R}^d$ be the feature for view $v$ and joint $j$ at time $t$. The pooled context is

$$
\bar{\mathbf{f}}_{t,j} = \frac{1}{V} \sum_{v=1}^{V} \mathbf{f}_{t,v,j},
$$

and the visibility logit is

$$
\hat{m}_{t,v,j} = \sigma\left( \text{MLP}\left([\mathbf{f}_{t,v,j}, \bar{\mathbf{f}}_{t,j}]\right) \right),
$$

where $\sigma$ is the sigmoid function.

To avoid degenerate triangulation when too few views are visible, a fallback guard forces all views to remain active if the predicted visible count for a joint is below `min_visible_views`:

```
visible = (visibility > threshold).float()
visible_count = visible.sum(dim=1)
fallback = (visible_count < min_visible_views)
effective_visibility = visibility + (1 - visibility) * fallback
```

This mechanism is implemented in `motionflow_mv/fusion/visibility_gated_fusion_v2.py` and integrated into `omniview_fusion_v2.py`, lines 203--228.

\subsection{Graph-Joint Attention for Skeleton-Aware Reasoning}

After intrinsic correction and per-frame ray-aware encoding, the model applies graph-joint attention over the product graph of views and joints. The implementation is in `motionflow_mv/fusion/graph_joint_attention_v2.py`.

The graph has one node per $(v, j)$ pair and four edge types:

- **Bone edges** (type 0): parent--child connections within each view.
- **Symmetry edges** (type 1): left--right mirror pairs within each view.
- **Cross-view edges** (type 2): same-joint connections across different views.
- **Self-loop edges** (type 3): identity edges.

For each layer, multi-head dot-product attention is computed over the directed edges. Edge-type embeddings are added to source values, and per-edge-type, per-head biases modulate attention scores. Normalization uses a stable scatter softmax over incoming edges for each destination node and head:

```
scores = (q_dst * k_src).sum(-1) / sqrt(head_dim) + edge_bias[edge_type]
attn = scatter_softmax(scores, dst)
out = index_add(attn * (v_src + edge_emb[edge_type]), dst)
```

This sparse, skeleton-aware attention is more anatomically grounded than dense joint self-attention and remains fully batched over the edge list, so it works for any view count. In `omniview_fusion_v2.py`, the graph block is inserted before the spatiotemporal transformer, lines 280--281.

\subsection{Bayesian Precision-Weighted Triangulation and Adaptive Gauss-Newton Refinement}

The central geometric step is a weighted Direct Linear Transform (DLT) triangulation. Each view and joint receives a learned precision weight derived from an anisotropic 2D covariance.

The covariance head outputs three raw parameters per $(v, j)$, which are converted to a lower-triangular Cholesky factor $\mathbf{L}_{t,v,j} \in \mathbb{R}^{2 \times 2}$:

$$
\boldsymbol{\Sigma}_{t,v,j} = \mathbf{L}_{t,v,j} \mathbf{L}_{t,v,j}^{\top}, \quad \mathbf{L}_{t,v,j} = \begin{bmatrix} l_{xx} & 0 \\ l_{xy} & l_{yy} \end{bmatrix}.
$$

The precision weight is

$$
w_{t,v,j}^{\text{prec}} = \frac{1}{\sqrt{\det \boldsymbol{\Sigma}_{t,v,j}}} = \frac{1}{l_{xx} \, l_{yy}}.
$$

The final scalar weight for each view and joint combines the predicted weight head output, detector confidence, visibility multiplier, and precision:

$$
w_{t,v,j} = \sigma\bigl(\text{weight\_head}(\mathbf{f})\bigr)_{v,j} \;\cdot\; c_{t,v,j} \;\cdot\; m_{t,v,j} \;\cdot\; w_{t,v,j}^{\text{prec}}.
$$

The weights are clamped to $w_{t,v,j} \ge 10^{-4}$ for numerical stability.

Given the corrected projection matrices $\mathbf{P}_v^{\text{corr}}$ and the 2D points, the weighted DLT solves, for each joint independently, the least-squares system

$$
\mathbf{X}_{t,j}^{\text{DLT}} = \arg\min_{\mathbf{X}} \sum_{v=1}^{V} w_{t,v,j} \, \bigl\| \mathbf{x}_{t,v,j} \times \mathbf{P}_v^{\text{corr}} \, [\mathbf{X}; 1] \bigr\|^2.
$$

This is implemented via a fully batched `torch.linalg.lstsq` call in `motionflow_mv/fusion/triangulation.py` and invoked in `omniview_fusion_v2.py`, line 317.

After DLT, adaptive Gauss-Newton refinement improves the estimate while remaining inside the camera model. A learned per-joint damping factor $\lambda_j$ is predicted from pooled features:

$$
\lambda_j = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \, \text{damping\_head}(\bar{\mathbf{f}}_{t,j}),
$$

with $\lambda_{\min}=10^{-6}$ and $\lambda_{\max}=10^{-2}$. The Gauss-Newton step is

$$
\mathbf{X}_{t,j}^{(k+1)} = \mathbf{X}_{t,j}^{(k)} - \left(\mathbf{J}^{\top} \mathbf{W} \mathbf{J} + \lambda_j \mathbf{I}\right)^{-1} \mathbf{J}^{\top} \mathbf{W} \, \mathbf{r}^{(k)},
$$

where $\mathbf{J}$ is the world-frame image Jacobian and $\mathbf{r}$ is the reprojection residual. The implementation runs for `gn_iters=2` steps and is defined in `ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`, lines 32--114.

\subsection{Residual Refinement and Training Objective}

Although the geometric pipeline is strong, small structured errors remain due to detector bias, residual calibration drift, and skeleton priors. A compact residual MLP adds a learned correction to the Gauss-Newton estimate:

$$
\mathbf{X}_{t,j}^{\text{final}} = \mathbf{X}_{t,j}^{\text{GN}} + \Delta \mathbf{X}_{t,j},
\quad \Delta \mathbf{X}_{t,j} = \text{MLP}\left([\bar{\mathbf{f}}_{t,j}, \mathbf{X}_{t,j}^{\text{GN}}]\right).
$$

The residual MLP has hidden dimension `residual_hidden=128` and is conditioned on the per-joint pooled feature and the geometric estimate. This is the final output of `omniview_fusion_v2.py`, lines 341--344.

The total training objective is

$$
\mathcal{L} = \mathcal{L}_{\text{3D}} + \lambda_{\text{epi}} \mathcal{L}_{\text{epi}} + \lambda_{\text{vis}} \mathcal{L}_{\text{vis}} + \lambda_{\text{pp}} \mathcal{L}_{\text{pp}} + \lambda_{\text{bone}} \mathcal{L}_{\text{bone}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}}.
$$

- $\mathcal{L}_{\text{3D}}$: Euclidean distance to 3D ground truth in millimetres.
- $\mathcal{L}_{\text{epi}}$: epipolar consistency weighted by predicted covariances (`epipolar_loss_weight=0.05`).
- $\mathcal{L}_{\text{vis}}$: binary cross-entropy against synthetic occlusion masks.
- $\mathcal{L}_{\text{pp}}$: principal-point offset regression against the inverse of the training perturbation.
- $\mathcal{L}_{\text{bone}}$: bone-length consistency regularizer.
- $\mathcal{L}_{\text{vel}}$: temporal velocity smoothness regularizer.

\subsection{HumanMotionIR and Downstream Consumption}

All upstream and downstream modules exchange a single intermediate representation, `HumanMotionIR`, with the following fields:

- **pose**: world-coordinate 3D joints $(T, J, 3)$ and optional SMPL parameters.
- **uncertainty**: per-frame, per-joint confidence, view support count, temporal alignment error, fusion disagreement, and optional scale uncertainty.
- **quality**: frame-valid flags and failure reasons.
- **provenance**: source manifest hash, per-view artifact hashes, GVHMR version, fusion plugin version, calibration hash, and IR builder version.

Before downstream consumption, the system passes quality gates for synchronization, calibration, occlusion, fusion disagreement, person association, motion continuity, and contact. Low-confidence frames can be rejected, flagged, or down-weighted in policy training.

Robot profiles externalize kinematics, retargeting, training, and export conventions. The same `HumanMotionIR` therefore drives different robots without retraining the fusion module.

\section{Training Details}

\subsection{Datasets}

We train and evaluate on non-circular benchmarks only:

- **Human3.6M true-GT**: 4 views, 17 joints. Standard protocol trains on subjects S1, S5, S6, S7, S8 and tests on S9 and S11. Labels are true mocap world coordinates from `data/h36m_true_gt/*_multiview_m.npz`. The old circular labels (`data/h36m_hf/`, `data/webbridge/h36m_meters/`) are excluded.
- **MPI-INF-3DHP**: up to 14 views, 28 joints. Train on subjects 1 and 3, validate on subject 2 sequence 1. The 2D inputs are detector outputs (RTMPose), not GT-projected 2D; detected-2D generation and DLT baseline validation are in progress.
- **AIST++ cross-domain smoke**: 9 views, 17 joints. Canonical `.npz` from `data/webbridge/aistpp_canonical/`, non-circular (DLT direct MJE $\approx$ 44 mm). Used to stress-test cross-domain transfer.
- **Shelf / Campus detected**: real 2D detections plus true 3D annotations from `data/webbridge/shelf_campus_detected/`. Campus (3 views) is the primary sparse-view benchmark.

\subsection{Augmentation}

Training-time camera perturbation is a core augmentation. Each clip receives independent noise on rotation, translation, focal length, and principal point. The current large-scale run uses principal-point perturbations of $\pm 5$ px and focal perturbations of $\pm 1\%$. Validation is always run on the unperturbed calibration.

Additional augmentations include 2D keypoint Gaussian noise, random joint occlusion, view dropout, and 2D outliers.

\subsection{Losses and Optimization}

The model is trained with AdamW. The current anchor configuration uses:

- Model: `RayAttentionFusionModelBayesianTriV2` / `OmniMultiViewFusionV2`
- Embedding dimension `d=128`
- Residual hidden dimension `residual_hidden=256`
- `n_st_layers=3`
- 50 epochs, 2,000 training samples, batch size 8
- Principal-point loss weight 0.2
- Epipolar loss weight 0.05

A staged warm-start strategy is used for `OmniMultiViewFusionV2`: load a Bayesian Tri v2 checkpoint, freeze the per-frame encoder and spatiotemporal transformer for 5 epochs, train only the new visibility, graph, and uncertainty heads, then unfreeze all and train end-to-end for 15--20 epochs.

\section{Inference and Evaluation Protocols}

\subsection{Metrics}

We report standard 3D human pose metrics:

- **MPJPE**: mean per-joint position error in millimetres.
- **PA-MPJPE**: MPJPE after Procrustes alignment.
- **PCK@50/100/150**: percentage of correct keypoints within 50, 100, and 150 mm.
- **AUC**: area under the PCK curve.
- **Reprojection error**: 2D reprojection residual after fusion.
- **Bone-length error**: deviation from the canonical skeleton.

\subsection{Robustness Protocol}

We evaluate under controlled corruptions:

- 2D keypoint Gaussian noise (e.g., 1.0 px, 2.0 px).
- Random joint occlusion (e.g., 20\% of joints).
- View dropout (e.g., 30\% of views).
- Calibration perturbations: rotation ($0.5^{\circ}$, $1.0^{\circ}$), translation (5 mm), focal length (1\%), principal point (3 px, 5 px).

\subsection{Runtime}

Inference throughput is measured on a single RTX 4090. The compact model runs at 12--195 clips/s depending on batch size and sequence length.

\section{Key Differences from Prior Work}

1. **Geometry-first decomposition.** Unlike end-to-end fusion methods that regress 3D joints directly, we keep triangulation at the center and learn only the structured error that geometry cannot fix: intrinsic correction, visibility, precision weights, and a small residual.

2. **Compact, robot-ready models.** The full model has well under 1.1 M parameters. Earlier MPI-INF-3DHP reports of 8.35 mm were obtained under circular GT-projected 2D protocols and are no longer valid. On the honest detected-2D protocol (RTMPose), DLT baselines are still being validated; the only available partial measurement on two files is ~150 mm and is not representative.

3. **Bayesian precision weighting.** Anisotropic image-space covariances give principled per-view uncertainty that feeds directly into weighted DLT, rather than using heuristic confidences alone.

4. **Adaptive Gauss-Newton refinement.** Refinement inside the camera model, with learned per-joint damping, keeps the solution metric and geometrically consistent.

5. **Plug-in architecture and HumanMotionIR.** The fusion module is exposed as a `MultiViewFusionPlugin` with explicit uncertainty and provenance, decoupling multi-view fusion from upstream single-view estimators and downstream robot retargeting.
