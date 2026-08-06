I researched the topic, read the relevant code and design docs, and drafted the report below. I could not write it to `docs/swarm_iter3/geometry_aware_attention.md` because this environment does not expose a file-writing tool and the subagent role is read-only for code, so the markdown should be saved by the parent/main agent.

# Geometry-Aware Attention for Calibrated Multi-View Human Pose Fusion

**Topic owner:** research swarm – Phase 1, Iteration 3  
**Target venues:** CVPR / ICRA 2027  
**Status:** exploratory; `attention_v2` prototype exists and is unstable.

---

## 1. Problem statement

In calibrated multi-view human pose estimation, the 3D skeleton is most often recovered by triangulating per-view 2D keypoints. The MotionFlow multi-view stack currently uses a confidence-weighted Direct Linear Transform (DLT) baseline that is deterministic, geometry-correct, and surprisingly hard to beat. Several lightweight learned fusion heads have been tried (`AttentionFusionModel`, `RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`), but on the real Shelf/Campus data they at best *match* DLT; none clearly surpass it.

The specific sub-problem this note addresses is:

> **How should camera geometry be injected into a learned attention fusion head so that attention scores become geometrically interpretable (e.g., down-weighting occluded or foreshortened views, exploiting epipolar consistency) rather than being learned as opaque scalar weights over 2D coordinates?**

Current `attention_v2` (`motionflow_mv/fusion/attention_model_v2.py`) naively flattens the 3×4 projection matrix into a 12-vector, embeds it with a linear layer, and adds it to the lifted 2D features before a per-joint softmax over views. On Shelf 300–600 this yields mean reprojection errors around **184 px** versus DLT's **~10 px** (README, Iteration 3). The model therefore fails to leverage the very geometric information that makes DLT successful.

A geometry-aware attention mechanism should:

1. **Preserve metric/geometric invariances** (scale, camera intrinsics, world coordinate frame).
2. **Make the attention score a function of viewing geometry** (ray directions, camera centers, epipolar relationships).
3. **Provide an inductive bias toward triangulation** so the network can reason in 3D, not just in image-plane correlations.
4. **Generalize across camera rigs** without dataset-specific re-tuning of projection-matrix embeddings.

---

## 2. Key related work / methods

### 2.1 Learnable triangulation and volumetric fusion

**Iskakov et al., "Learnable Triangulation of Human Pose," ICCV 2019** [1]
- Proposes *algebraic* triangulation with learned per-view weights and *volumetric aggregation* with 3D CNNs.
- The algebraic variant predicts view weights and combines them in a weighted DLT formulation—very close in spirit to the current `RobustTriangulationModel`.
- Takeaway for us: a learned weighting head is useful, but the *triangulation step itself* must remain explicit and differentiable.

### 2.2 Geometry-biased / geometry-aware transformers

**Moliner et al., "Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction," arXiv 2312.17106 / 2024** [2]
- Adds geometric biases to multi-head self-attention: attention scores are conditioned on epipolar geometry and camera-ray information.
- Specifically, pairwise attention between views is modulated by geometric compatibility (e.g., rays from different views should intersect at the true 3D point).
- Takeaway: geometry should enter the *attention score*, not just the feature vector.

### 2.3 Multi-view geometry transformers (MVGFormer)

**Liao et al., "Multiple View Geometry Transformers for 3D Human Pose Estimation," CVPR 2024** [3]
- Interleaves learning-free geometry modules with learnable appearance/geometry transformers.
- Projects image features onto epipolar lines and uses geometry-aware cross-view attention to collect evidence.
- Takeaway: the strongest architectures are *hybrids*—geometry modules provide constraints, transformers handle noise and occlusion.

### 2.4 State-space and efficient multi-view scanning

**Chharia et al., "MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation," CVPR 2025** [4]
- Replaces dense O(V²) cross-view attention with directed state-space scanning across camera rays.
- Motivation: dense attention over views is expensive and can overfit to a particular camera layout.
- Takeaway: even if we start with dense attention, designing geometry-aware *sparse* or *ray-ordered* attention is a scalable direction.

### 2.5 Skeleton-level temporal/physical priors

**Bragagnolo et al., "Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation," ECCVW 2024** [5]
- Fuses monocular 3D skeletons with reprojection, bone-length, and limb-symmetry constraints.
- Shows that physical/skeletal priors can outperform pure triangulation under occlusion.
- Takeaway: geometry-aware attention should eventually be combined with bone-length and temporal losses, not trained on reprojection alone.

---

## 3. How it relates to the current MotionFlow multi-view codebase

### 3.1 Existing fusion plugin hierarchy

The current architecture is clean and plugin-based (`motionflow_mv/fusion/fusion_module.py`):

- `DLTFusion`: confidence-weighted DLT, the geometric baseline.
- `AttentionFusionModule` (v1): per-joint view attention over lifted `(x, y, c)` features; ignores cameras.
- `AttentionFusionV2Module` (v2): adds a flattened 12-D projection matrix per view.
- `RobustTriangulationFusion`: learns per-view weights and solves a differentiable DLT.
- `ResidualRefinerFusion`: DLT + learned residual correction.
- `TemporalRefinerFusion`: Bi-GRU temporal smoothing over DLT poses.

### 3.2 Why `attention_v2` is unstable

`AttentionFusionModelV2` (`motionflow_mv/fusion/attention_model_v2.py`) does:

```python
x = self.lift(x)                       # (B, V, J, D) from 2D+confidence
cam = self.cam_embed(cameras)          # (B, V, D) from flattened P
x = x + cam                            # element-wise add, broadcast over joints
```

Problems:

1. **No scale normalization.** Projection matrices in Shelf are in millimeters, while synthetic data may be in meters. A single linear layer on raw `P` conflates focal length, translation, and world scale.
2. **No per-joint ray information.** The same camera embedding is added to every joint, so the model cannot express “view *v* sees joint *j* along ray *r*”.
3. **No triangulation inductive bias.** The head directly regresses `(B, J, 3)`; there is no differentiable triangulation layer forcing the network to respect multi-view geometry.
4. **Training target is DLT pseudo-GT.** The model learns to reproduce a noisy geometric baseline rather than true 3D coordinates.

These issues are consistent with the observed numbers in `docs/design_v3.md` and `README.md`: v2 performs worse than v1, and both lag far behind DLT.

### 3.3 Natural integration points

A geometry-aware attention module fits into the existing stack in three ways:

1. **Replace the v2 head.** Keep the `FusionModule` interface, but change `AttentionFusionModelV2.forward` to use ray embeddings and a differentiable triangulation layer.
2. **Augment `RobustTriangulationModel`.** The per-view weight head already exists; make its attention *geometry-aware* by feeding ray-direction and epipolar embeddings.
3. **Plug into the IR adapter.** `motionflow_mv/ir/multiview_adapter.py::fuse_multiple_irs` already produces per-view 2D + confidence and calls a `FusionModule`. Any improved geometry-aware plugin drops in without changing the IR contract.

---

## 4. Concrete recommendations

### 4.1 Short-term: stabilize `attention_v2` with ray-direction embeddings

Implement a normalized *ray representation* for each `(view, joint)` pair:

```
c_v = -R_v^T t_v                         # camera center in world
r_vj = (X_ref - c_v) / ||X_ref - c_v||    # approximate ray direction to joint
```

where `X_ref` can be the DLT triangulated point or the centroid of all view rays. Embed the pair `(r_vj, c_v)` instead of the raw flattened `P`. This removes scale ambiguity and makes attention scores depend on actual viewing geometry.

**Files to change:**
- `motionflow_mv/fusion/attention_model_v2.py`
- `motionflow_mv/fusion/attention_fusion_v2_module.py`
- `experiments/train_attention_fusion_shelf_v2.py` (input normalization)

### 4.2 Mid-term: geometry-biased attention score

Modify `ViewAttentionFusion` so the score between a query joint and a view includes a geometric compatibility term:

```python
scores = (k_vj^T q_j) / sqrt(D) + beta * g(ray_vj, camera_v)
```

where `g` can be:

- **Epipolar compatibility** with another reference view: how close the ray from view *v* lies to the epipolar line of view *u*.
- **Baseline/angle feature**: encode the angle between camera baseline and ray.
- **Occlusion/foreshortening prior**: views with rays nearly parallel to the limb or body surface should be down-weighted.

This directly draws from the Geometry-Biased Transformer [2] and MVGFormer [3].

### 4.3 Add a differentiable triangulation layer as an inductive bias

Instead of regressing 3D coordinates end-to-end, the attention module should predict **per-view weights** (or refined 2D offsets) and feed them into a differentiable DLT solver, similar to `RobustTriangulationModel` (`motionflow_mv/fusion/robust_triangulation.py`).

Advantages:
- The network learns *which views to trust*, while geometry guarantees a physically consistent 3D point.
- Training on pseudo-GT becomes much more stable because the final layer is geometry-constrained.
- Cross-dataset generalization improves because weights are learned over normalized rays, not raw pixel coordinates.

### 4.4 Training recipe

1. **Normalize everything to meters.** The IR already uses meters; camera parameters should be divided by the dataset unit (Shelf uses mm → divide by 1000).
2. **Use real 3D GT.** Pseudo-GT from DLT caps performance at the DLT level. Human3.6M is the most practical source for supervised 3D training; Campus/Shelf can remain reprojection sanity sets.
3. **Add geometric/physical losses.**
   - `L_MPJPE` (3D supervised)
   - `L_reproj` (reprojection consistency)
   - `L_bone` (bone-length prior)
   - `L_sym` (limb symmetry)
   - `L_temp` (temporal smoothness)
4. **Synthetic pre-training on AMASS.** Render SMPL motion through virtual calibrated rigs with realistic 2D detection noise and occlusion, then fine-tune on real data. This is already outlined in `docs/design_v2.md` Section 4.2.

### 4.5 Suggested paper angle

Frame the contribution as a **geometry-aware attention fusion plugin** within a modular multi-view pipeline, not as “a new triangulation algorithm.” Potential titles:

- *RayFormer: Ray-Guided Attention for Calibrated Multi-View Human Pose Estimation*
- *Geometry-Biased View Attention for Multi-View Human Motion Fusion*
- *Plug-in Geometry-Aware Fusion for World-Coordinate Human Motion IR*

Key claims:
1. A unified `FusionModule` plugin interface where DLT and learned attention are interchangeable.
2. A ray/epipolar embedding that makes view attention invariant to camera intrinsics and world scale.
3. Empirical gains on Human3.6M + Shelf/Campus, with DLT as a strong geometric baseline.
4. Uncertainty-aware `HumanMotionIR` populated with per-view attention weights and reprojection residuals (robotics/replanning angle for ICRA).

---

## 5. Open questions / risks

| Risk / Question | Why it matters | Mitigation |
|-----------------|----------------|------------|
| **3D ground-truth access** | Without real 3D GT, learned fusion cannot clearly beat DLT. | Register Human3.6M; use AMASS synthetic pre-training as a fallback. |
| **`attention_v2` instability** | Naive projection-matrix embedding is scale-sensitive and under-constrained. | Switch to normalized ray/camera-center embeddings and a triangulation layer. |
| **Cross-dataset camera layouts** | Attention learned on one rig may not transfer to another. | Use camera-invariant ray features and evaluate zero-shot on Campus/Shelf. |
| **DLT is a strong baseline** | Any learned method must show a statistically significant margin. | Report MPJPE, PA-MPJPE, PCK, and per-joint breakdowns; not just reprojection. |
| **Computational cost** | Dense cross-view attention scales as O(V²); future rigs may have many cameras. | Adopt sparse/ray-ordered attention (MV-SSM [4]) after the dense baseline works. |
| **Calibration sensitivity** | All methods assume accurate intrinsics/extrinsics. | Keep DLT as the default fallback; add robust calibration (COLMAP/DUSt3R) later. |
| **SMPL fitting vs. skeleton** | Current IR supports SMPL pose; fusion is currently on 3D joints. | Extend to SMPL-aware fusion once joint-level geometry-aware attention is validated. |

---

## 6. Bottom line

The current `attention_v2` attempt is a useful prototype that proves the need for geometry-aware attention, but its naive projection-matrix embedding is unstable and underperforms DLT. The next step is to **replace the raw 12-D projection embedding with normalized ray/camera-center features, bias the attention score with epipolar or viewing-angle terms, and tie the network to a differentiable triangulation layer**. Combined with 3D-supervised training on Human3.6M and synthetic pre-training on AMASS, this is the most credible path toward a CVPR/ICRA 2027 contribution that can genuinely surpass the DLT baseline while remaining a drop-in plugin in the existing MotionFlow multi-view stack.

---

## References

1. I. Iskakov et al., “Learnable Triangulation of Human Pose,” *ICCV*, 2019.
2. A. Moliner et al., “Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction,” *arXiv:2312.17106*, 2023/24.
3. S. Liao et al., “Multiple View Geometry Transformers for 3D Human Pose Estimation,” *CVPR*, 2024.
4. R. Chharia et al., “MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation,” *CVPR*, 2025.
5. M. Bragagnolo et al., “Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation,” *ECCVW*, 2024.