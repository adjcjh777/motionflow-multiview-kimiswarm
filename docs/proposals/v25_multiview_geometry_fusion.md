# v25: Multi-View Geometry Fusion (GeoMV Fusion)

**Task identifier:** `design_v25_multiview_geometry_fusion`  
**Status:** Design / Prototype  
**Depends on:** v17 (`cross_view_transformer_v17`), v18 (`deformable_cross_view_attention`), v21 (`neural_bundle_adjustment_v21`), v22 (`kinematic_anthropometric_prior_v22`)

> **Versioning note:** This proposal reclaims the `v25` label for a
> camera/geometry-centric fusion model. The earlier temporal-only draft
> `docs/proposals/v25_temporal_fusion.md` is renumbered to
> `docs/proposals/v26_temporal_fusion.md`; its ideas are absorbed here as the
> optional *temporal geometry attention* component (Section 3.5).

## 1. Motivation

The current best stack in `OmniMultiViewFusionV5` combines several complementary
ideas:

* **v18** sparse cross-view attention guided by epipolar geometry.
* **v21** neural bundle adjustment that jointly refines 3D joints and cameras.
* **v22** kinematic anthropometric prior (KAP) that regularises bone lengths and
  joint angles.

Despite these additions, geometry is still used *indirectly*: attention scores
are biased by epipolar distance, and camera updates are predicted by a small MLP
from reprojection statistics. The v21 neural BA regression to 128.27 mm on
WebBridge shows that a neural camera-correction head can diverge when it is not
anchored by explicit geometric constraints.

**v25** therefore redesigns the multi-view fusion core so that the model reasons
*explicitly* with cameras and 3D geometry throughout the forward pass:

* Tokens are built from **viewing rays** (camera centre + ray direction), not
  just from 2D coordinates.
* Cross-view attention scores are conditioned on **ray intersection quality** and
  **baseline angle**, not only epipolar distance.
* Triangulation is upgraded to a **learned depth-proposal head** that reasons
  about per-ray depth hypotheses before fusing views.
* A **geometry bundle-adjustment (GeoBA)** block refines both 3D structure and
  cameras using analytic reprojection, epipolar, and cheirality constraints,
  with bounded updates initialised to identity.
* An optional **camera-joint graph** propagates multi-view constraints across the
  skeleton.

## 2. Design principles

1. **Geometry first.** Every multi-view operation should be expressible in terms
   of rays, camera centres, reprojection, and 3D distance.
2. **Bounded, warm-startable updates.** Any camera or pose refinement block
   must start as the identity map and apply small, clamped updates, to avoid the
   v21-style regression.
3. **Modular drop-in.** v25 is a single module with toggles; it can be stacked
   on top of the existing v18 + v22 pipeline that is currently running as v23 and
   v24.
4. **Variable-view compatible.** Ray tokenisation and geometry attention must
   accept a `view_mask` and an arbitrary number of views.
5. **Supervision by both 3D and geometry losses.** In addition to MPJPE, v25 is
   trained with reprojection, epipolar, and cheirality losses.

## 3. Module overview

**File:** `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`

```text
MultiViewGeometryFusionV25(
    d: int = 128,
    n_heads: int = 4,
    n_views: int = 4,
    n_geometry_layers: int = 2,
    n_ray_samples: int = 4,
    use_geometry_attention: bool = True,
    use_learned_depth_triangulation: bool = True,
    use_geometry_bundle_adjustment: bool = True,
    use_camera_joint_graph: bool = False,
    max_camera_rotation_deg: float = 2.0,
    max_camera_translation: float = 0.1,
    max_focal_scale: float = 0.05,
    max_principal_point_px: float = 10.0,
    max_point_update_m: float = 0.05,
    dropout: float = 0.1,
)
```

### 3.1 Inputs / outputs

**Forward signature**

```python
pred_3d_ref, K_ref, R_ref, t_ref, geom_loss = geom_fusion(
    feat,            # (B, T, V, J, d)   current per-view feature tokens
    points_2d,       # (B, T, V, J, 2)   detected 2D keypoints
    K,               # (B, T, V, 3, 3)   intrinsics
    R,               # (B, T, V, 3, 3)   rotations
    t,               # (B, T, V, 3)      translations
    pred_3d_init,    # (B, T, J, 3)      initial triangulated pose
    view_mask,       # (B, T, V)         optional view mask
)
```

**Outputs**

* `pred_3d_ref`: `(B, T, J, 3)` — refined 3D pose.
* `K_ref`: `(B, T, V, 3, 3)` — refined intrinsics (identity if GeoBA disabled).
* `R_ref`: `(B, T, V, 3, 3)` — refined rotations.
* `t_ref`: `(B, T, V, 3)` — refined translation.
* `geom_loss`: scalar — sum of geometry-aware auxiliary losses.

### 3.2 Ray tokenisation

For each view `v` and joint `j` we build a ray token in world coordinates:

```
c_v    = -R_v^T t_v                                    # camera centre
d_vj   = R_v^T K_v^{-1} [u_vj, v_vj, 1]^T             # world ray direction (normalised)
ray_vj = MLP([ d_vj ; c_v ; conf_vj ; z_vj ] ; d)    # d-dimensional token
```

where `conf_vj` is the 2D detection confidence and `z_vj` is a learned depth
embedding. The depth embedding is parameterised as a small lookup table of
`n_ray_samples` depth hypotheses, projected to 3D points along the ray and fed
through a 1D conv.

Key property: because the ray token is defined in world space, the model can
fuse information across arbitrary camera setups without learned positional
embeddings.

### 3.3 Geometry-aware cross-view attention

The generic cross-view attention in v5/v17 is replaced (or augmented) by a
geometry-aware attention head operating on ray tokens.

For a query ray `(v_q, j)` and a key ray `(v_k, j)` we compute three additive
logits:

1. **Content logit** (standard scaled dot-product on the projected ray
   features):
   ```
   logit_content = (Q_vq · K_vk) / sqrt(d_h)
   ```

2. **Epipolar logit** from the v18/v5 epipolar bias, measuring how close the key
   ray is to the epipolar plane induced by the query ray.

3. **Ray-intersection logit**, derived from the shortest distance between the
   two rays in 3D and the cosine of their baseline angle:
   ```
   logit_ray = - ( ray_dist(v_q, v_k) / sigma_d + (1 - cosθ) / sigma_a )
   ```
   where `σ_d` and `σ_a` are learnable temperature parameters.

The final attention score is:

```
logit = logit_content + logit_epipolar + logit_ray
```

Masked views are excluded and the output is a residual added to the original
feature tokens.

### 3.4 Learned depth-proposal triangulation head

Instead of a single DLT solve followed by a residual MLP, v25 introduces a
**DepthProposalTriangulation** head:

1. For each view/joint ray, sample `n_ray_samples` depths `z ∈ [z_min, z_max]`.
2. Project each depth to a 3D candidate:
   ```
   X_vj^k = c_v + z_k * d_vj
   ```
3. Score each candidate using an attention mechanism that compares it to the
   candidates from the other views. The score is conditioned on the epipolar and
   ray-intersection quality.
4. Aggregate to a single 3D point per joint:
   ```
   X_j = sum_v w_v * X_vj,   w_v = softmax_v(score_v)
   ```

This head is supervised by the ground-truth 3D position and, indirectly, by the
reprojection loss on the refined cameras.

### 3.5 Geometry bundle adjustment (GeoBA)

GeoBA is a differentiable refinement block with a fixed number of iterations
(default `n_iters=2`). It refines the initial triangulation and the camera
parameters.

**Structure step** — one damped Gauss-Newton/Levenberg-Marquardt step on the
reprojection error:

```
(J^T W J + λ I) ΔX = J^T W r
```
where `r` is the 2D reprojection residual and `W` are the learned view weights.
The update `ΔX` is clamped to `[-max_point_update_m, max_point_update_m]`.

**Camera step** — a lightweight MLP predicts a bounded camera correction from the
same reprojection residual **plus** the current ray-intersection quality and the
per-view depth consistency. The predicted update is bounded exactly as in v21:

```
K' = K * (1 + tanh(df) * max_focal_scale)
R' = R_delta * R
 t' =  t + tanh(dt) * max_camera_translation
```

The final layer of the camera head is initialised to zero, so the block starts
as an identity/no-op.

**Geometry losses inside GeoBA**

* **Reprojection loss:** `|| P_v X_j - x_vj ||_2` weighted by visibility.
* **Epipolar loss:** existing `epi_loss` from `omniview_fusion_v5`.
* **Cheirability loss:** penalise points that fall behind any camera or too
  close to a camera centre:
  ```
  L_cheir = sum max(0, -depth) + max(0, z_near / depth - 1)
  ```
* **Depth-consistency loss:** encourage the per-view depth proposals to agree
  with the fused 3D point.

### 3.6 Optional camera-joint graph refinement

When `use_camera_joint_graph=True`, a small bipartite graph neural network is
built:

* Nodes: `V` camera nodes and `J` joint nodes.
* Edges: a camera `v` is connected to joint `j` if the joint is visible in view
  `v`.
* Edge features: ray direction, reprojection residual, and inverse depth.

A few message-passing steps (default `n_gnn_layers=2`) refine both camera and
joint features; these are then used by GeoBA in the next iteration.

### 3.7 Temporal geometry attention (absorbed from the old v25 temporal draft)

The temporal extension is kept as an optional submodule. For each query frame,
the model attends to a small set of temporal neighbours (`Δt ∈ {-1, 0, +1}`)
using the same ray-intersection and epipolar logits. This adds temporal context
before triangulation at a cost linear in `T`.

## 4. Integration into `OmniMultiViewFusionV5`

### 4.1 New toggles

```python
use_multiview_geometry_fusion_v25: bool = False,
v25_use_geometry_attention: bool = True,
v25_use_learned_depth_triangulation: bool = True,
v25_use_geometry_bundle_adjustment: bool = True,
v25_use_camera_joint_graph: bool = False,
v25_geom_loss_weight: float = 0.1,
```

### 4.2 Instantiation

```python
self.use_multiview_geometry_fusion_v25 = use_multiview_geometry_fusion_v25
if self.use_multiview_geometry_fusion_v25:
    self.multiview_geometry_fusion_v25 = MultiViewGeometryFusionV25(
        d=d,
        n_heads=n_heads,
        n_views=n_views,
        use_geometry_attention=v25_use_geometry_attention,
        use_learned_depth_triangulation=v25_use_learned_depth_triangulation,
        use_geometry_bundle_adjustment=v25_use_geometry_bundle_adjustment,
        use_camera_joint_graph=v25_use_camera_joint_graph,
    )
```

### 4.3 Forward pass hook

The module is inserted **after** the v18 deformable cross-view attention block
and **before** the spatio-temporal (ST) transformer. This lets the model use
camera/geometry reasoning early, before the global ST transformer mixes
temporal information.

```python
# v18 sparse cross-view attention (per-frame).
if self.use_deformable_cross_view_attention_v18 and ...:
    feat = self.deformable_cross_view_attention_v18(...)

# v25 geometry fusion.
if self.use_multiview_geometry_fusion_v25 and ...:
    pred_3d_init = ... # current triangulated pose before GN refinement
    pred_3d_ref, K_ref, R_ref, t_ref, geom_loss = \
        self.multiview_geometry_fusion_v25(
            feat, points_2d, K_corrected, R, t,
            pred_3d_init, view_mask,
        )
    K_corrected, R, t = K_ref, R_ref, t_ref
    epi_loss = epi_loss + self.v25_geom_loss_weight * geom_loss
```

After v25, the pipeline continues with the ST transformer, the existing
Gauss-Newton refinement, the residual/diffusion head, KAP (v22), and the v19
temporal perceiver.

## 5. Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | 128 | Feature dimension (matches v5). |
| `n_heads` | 4 | Attention heads in geometry attention. |
| `n_geometry_layers` | 2 | Number of stacked geometry attention layers. |
| `n_ray_samples` | 4 | Depth hypotheses per ray for learned triangulation. |
| `max_camera_rotation_deg` | 2.0 | Maximum camera rotation correction per GeoBA iteration. |
| `max_camera_translation` | 0.1 | Maximum camera translation correction (m). |
| `max_focal_scale` | 0.05 | Maximum multiplicative focal-length correction. |
| `max_principal_point_px` | 10.0 | Maximum principal-point correction (pixels). |
| `max_point_update_m` | 0.05 | Maximum 3D point update per GeoBA iteration. |
| `v25_geom_loss_weight` | 0.1 | Weight of the auxiliary geometry losses. |

## 6. Losses

In addition to the existing losses, v25 contributes:

| Loss | Source | Weight |
|------|--------|--------|
| Reprojection | GeoBA refined cameras + points | `reproj_loss_weight` |
| Epipolar | Existing `epi_loss` | `epipolar_loss_weight` |
| Cheirability | Positive-depth constraint | `v25_geom_loss_weight * 0.1` |
| Depth consistency | Per-view depth proposals vs fused point | `v25_geom_loss_weight * 0.2` |
| MPJPE | Final refined 3D pose | supervised directly |

The camera-correction head and the depth-proposal head are trained jointly with
the rest of the network; no separate stage is required.

## 7. Test coverage

Add `tests/test_multiview_geometry_fusion_v25.py` covering:

* Forward shape `(B, T, V, J, 3) → (B, T, J, 3)`.
* Camera matrices remain valid (`det(R) ≈ 1`, `K` upper triangular).
* Gradient flow through `feat`, `points_2d`, `K`, `R`, `t`, and `pred_3d_init`.
* GeoBA is identity at init: `||pred_ref - pred_init||` small.
* Cheirability loss is zero for points in front of all cameras.
* View masking: masked views do not contribute to geometry attention.
* Works for `J=17` and `J=28` skeletons.

Also add a toggle-on case to `tests/test_omniview_fusion_v5.py`.

Run:

```bash
pytest tests/test_multiview_geometry_fusion_v25.py tests/test_omniview_fusion_v5.py -q
```

## 8. Relation to running A800 experiments

* **v23** (`v18 + KAP`, no BA) is the safest warm-start checkpoint for v25.
  Enable `use_multiview_geometry_fusion_v25=True` while keeping
  `use_neural_bundle_adjustment_v21=False`; the geometry attention and depth
  head start as identity, and GeoBA starts with zero camera correction.
* **v24** (`v18 + fixed BA + KAP`) already has a bounded v21 camera head.
  v25 supersedes that block: disable `use_neural_bundle_adjustment_v21` and use
  `v25_use_geometry_bundle_adjustment=True` instead.
* **v21 standalone** should not be used together with GeoBA to avoid redundant
  and possibly conflicting camera updates.

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Extra compute from ray tokenisation and depth sampling. | Cache ray directions; use only `n_ray_samples=4` at first. |
| GeoBA camera correction diverges like v21. | Identity initialisation, small bounds, and clamped point updates. |
| Learned depth head overfits to camera setup. | Strong data augmentation of camera intrinsics/extrinsics. |
| Geometry losses dominate over MPJPE. | Start with `v25_geom_loss_weight=0.1` and tune. |
| Conflicts with v22 KAP. | Apply KAP after v25, as v22 already does after triangulation. |

## 10. Future work

* Learn continuous depth proposals instead of a fixed grid.
* Replace the depth-proposal head with a tiny neural radiance-style field for
  arbitrary camera positions.
* Fuse the camera-joint graph with the existing skeleton graph attention in v5.
* Use v25 as a teacher in a distillation pipeline to obtain a single-view
  geometry-aware student.

## 11. References

* `motionflow_mv/fusion/omniview_fusion_v5.py`
* `motionflow_mv/fusion/deformable_cross_view_attention.py`
* `motionflow_mv/fusion/neural_bundle_adjustment_v21.py`
* `motionflow_mv/fusion/differentiable_bundle_adjustment.py`
* `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py`
* `motionflow_mv/fusion/epipolar_transformer_bias.py`
* `motionflow_mv/fusion/camera_conditioned_view_embedding.py`
