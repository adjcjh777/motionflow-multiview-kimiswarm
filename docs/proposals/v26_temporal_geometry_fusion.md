# v26: Temporal Multi-View Geometry Fusion

**Task identifier:** `design_v26_temporal_geometry_fusion`  
**Depends on:** v25 (`docs/proposals/v25_multiview_geometry_fusion.md`)

## 1. Motivation

v25 introduced `MultiViewGeometryFusionV25` — a per-frame geometry fusion block
that reasons with rays, epipolar constraints and learned depth proposals. It
improved multi-view consistency but still processes every frame in isolation.

v26 extends v25 temporally: each (time, view, joint) token now attends to a
small window of neighbouring frames across all views. This adds explicit
temporal reasoning **before** the final triangulation, at a cost linear in the
temporal window size, while preserving the geometry-first design of v25.

Key differences from existing temporal components:

| Component | Level | What it fuses | Cost |
|-----------|-------|---------------|------|
| v19 Temporal Perceiver | Post-triangulation 3D poses | Whole clip → latent set | `O(T · J · n_latents)` |
| v25 Multi-View Geometry Fusion | Per-frame rays before triangulation | Views within one frame | `O(V² · J)` |
| **v26 Temporal Geometry Fusion** | Spatio-temporal rays before triangulation | Small temporal window × views | `O(T · V² · W · J)` |

Because `W` is small (default `3`), v26 is a lightweight way to inject temporal
context without the full quadratic cost of the spatio-temporal transformer.

## 2. Design principles

1. **Geometry-first, like v25.** Temporal fusion is still expressed in terms of
   rays, epipolar distance and ray-intersection quality.
2. **Bounded, warm-startable updates.** The temporal attention output projection
   and temporal positional bias are initialised near zero, so the block starts
   as an identity residual.
3. **Modular drop-in.** v26 re-uses the v25 ray tokenizer, geometry attention
   and depth-proposal triangulation. It can replace v25 in the v5 pipeline with
   an single toggle.
4. **Variable-view compatible.** View masks are propagated through the temporal
   window.
5. **Testable and small.** The first prototype fits in one module and one test
   file.

## 3. Module overview

**File:** `motionflow_mv/fusion/temporal_geometry_fusion_v26.py`

```text
TemporalGeometryFusionV26(
    d: int = 128,
    n_heads: int = 4,
    n_views: int = 4,
    n_geometry_layers: int = 2,
    n_temporal_layers: int = 1,
    n_ray_samples: int = 4,
    temporal_window: int = 3,
    use_geometry_attention: bool = True,
    use_temporal_geometry_attention: bool = True,
    use_learned_depth_triangulation: bool = True,
    use_temporal_depth_consistency: bool = False,
    temporal_loss_weight: float = 0.1,
    dropout: float = 0.1,
)
```

### 3.1 Inputs / outputs

**Forward signature** (drop-in replacement for v25):

```python
pred_3d_ref, geom_loss = temporal_geom_fusion(
    points_2d,       # (B, T, V, J, 2/3) detected 2D keypoints (+ optional confidence)
    K,               # (B, T, V, 3, 3) intrinsics
    R,               # (B, T, V, 3, 3) rotations
    t,               # (B, T, V, 3) translations
    pred_3d_init,    # (B, T, J, 3) initial triangulated pose (optional)
    view_mask,       # (B, T, V) optional view mask
    confidence,      # (B, T, V, J) optional confidence weights
)
```

**Outputs**

* `pred_3d_ref`: `(B, T, J, 3)` — refined 3D pose.
* `geom_loss`: scalar — sum of reprojection loss and optional temporal
  smoothness loss.

### 3.2 Temporal geometry attention

**File:** same module, class `TemporalGeometryAttention`.

For each joint, tokens are arranged as a sequence over `(time, view)` pairs.
For a query `(t, v_q)`, keys are gathered from frames
`[t - half_window, ..., t + half_window]` across all views.

The attention score is the sum of four additive terms:

```
logit = logit_content / sqrt(d_h)
        - epipolar_dist[t, v_q, v_k]            # v25 bias
        + ray_logit[t, v_q, v_k]                 # v25 ray-intersection quality
        + temporal_pos[dt]                       # learnable temporal offset
```

Out-of-bound temporal offsets are masked. View masking is applied to every
frame in the temporal window.

**Complexity:** per layer `O(B · T · V² · W · J)`, where `W = temporal_window`
(default `3`). Memory is dominated by the `(T, V) × (W, V)` attention logits per
joint, which is tiny for the default settings.

### 3.3 Re-using v25 components

v26 imports and reuses from `multiview_geometry_fusion_v25.py`:

* `compute_rays`
* `ray_intersection_logit`
* `RayTokenizer`
* `GeometryAwareCrossViewAttention`
* `DepthProposalTriangulation`
* `triangulate_initial`

This keeps the implementation minimal and guarantees that disabling the new
temporal path reproduces v25 behaviour.

### 3.4 Temporal smoothness loss

When `use_temporal_depth_consistency=True` and `T > 1`, a velocity-smoothness
term is added to the geometry loss:

```
L_temporal = mean(|| pred_3d_ref[:, t] - pred_3d_ref[:, t-1] ||_2)
L_geom = L_reproj + temporal_loss_weight * L_temporal
```

This is a placeholder for a richer temporal depth-consistency loss; a future
iteration may regularise per-ray depth proposals instead of the final 3D
points.

## 4. Integration into `OmniMultiViewFusionV5`

### 4.1 New toggles

```python
use_temporal_geometry_fusion_v26: bool = False,
v26_use_geometry_attention: bool = True,
v26_use_temporal_geometry_attention: bool = True,
v26_use_learned_depth_triangulation: bool = True,
v26_use_temporal_depth_consistency: bool = False,
v26_temporal_loss_weight: float = 0.1,
v26_temporal_window: int = 3,
v26_geom_loss_weight: float = 0.1,
```

### 4.2 Instantiation

```python
self.use_temporal_geometry_fusion_v26 = use_temporal_geometry_fusion_v26
self.v26_geom_loss_weight = v26_geom_loss_weight
if self.use_temporal_geometry_fusion_v26:
    from motionflow_mv.fusion.temporal_geometry_fusion_v26 import TemporalGeometryFusionV26
    self.temporal_geometry_fusion_v26 = TemporalGeometryFusionV26(
        d=self.d,
        n_heads=self.n_heads,
        n_views=n_views,
        use_geometry_attention=v26_use_geometry_attention,
        use_temporal_geometry_attention=v26_use_temporal_geometry_attention,
        use_learned_depth_triangulation=v26_use_learned_depth_triangulation,
        use_temporal_depth_consistency=v26_use_temporal_depth_consistency,
        temporal_loss_weight=v26_temporal_loss_weight,
        temporal_window=v26_temporal_window,
    )
```

### 4.3 Forward pass hook

Replace the existing v25 hook in `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
# v26 temporal geometry fusion refinement (replaces v25).
if self.use_temporal_geometry_fusion_v26 and self.temporal_geometry_fusion_v26 is not None:
    pred_3d_gn_v26, geom_loss_v26 = self.temporal_geometry_fusion_v26(
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        pred_3d_init=pred_3d_gn.view(B, T, J, 3),
        view_mask=view_mask_flat.view(B, T, V),
        confidence=confidences.view(B, T, V, J),
    )
    pred_3d_gn = pred_3d_gn_v26.view(B * T, J, 3)
    geom_loss_v25 = geom_loss_v26
```

This is a direct replacement: v26 subsumes the per-frame v25 block and adds
temporal context.

## 5. Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | `128` | Feature dimension (matches v5). |
| `n_heads` | `4` | Attention heads. |
| `n_geometry_layers` | `2` | Per-frame geometry-attention layers. |
| `n_temporal_layers` | `1` | Spatio-temporal attention layers. |
| `n_ray_samples` | `4` | Depth hypotheses per ray. |
| `temporal_window` | `3` | Temporal window size (`[-1, 0, +1]`). |
| `temporal_loss_weight` | `0.1` | Weight of the temporal smoothness loss. |
| `v26_geom_loss_weight` | `0.1` | Weight of the v26 auxiliary geometry loss. |

## 6. Losses

In addition to the existing v25 losses, v26 contributes:

| Loss | Source | Weight |
|------|--------|--------|
| Reprojection | v26 refined cameras + points | `reproj_loss_weight` |
| Temporal smoothness | 3D point velocity | `temporal_loss_weight` |
| MPJPE | Final refined 3D pose | supervised directly |

## 7. Test coverage

Tests are in `tests/test_temporal_geometry_fusion_v26.py` and cover:

* Forward shape `(B, T, V, J, 3) → (B, T, J, 3)` for `J ∈ {17, 28}`.
* `TemporalGeometryAttention` output shape equals input shape.
* Identity-at-init when all learned blocks are disabled.
* Gradient flow through `points_2d`, `K`, `R`, `t` and `pred_3d_init`.
* View masking: dropped views do not contribute.
* Temporal boundary handling for `T < temporal_window` and `T ≥ temporal_window`.
* Toggle coverage across all combinations of geometry/temporal/depth toggles.
* Invalid head dimensions and even `temporal_window` raise `ValueError`.

Run:

```bash
pytest tests/test_temporal_geometry_fusion_v26.py -q
```

## 8. Relation to running A800 experiments

* **v25 small** is currently training with v18 + v25. The v26 prototype is a
  drop-in replacement for the v25 module; a first smoke test can be launched
  locally on the RTX 4090 before touching A800.
* **Warm start**: because the temporal attention output projection is
  initialised to zero, a v25 checkpoint can be loaded into the v26 module with
  `use_temporal_geometry_attention=True` and the model starts from v25
  behaviour, then learns to use temporal context.
* **GPU cost**: the extra compute over v25 is the spatio-temporal attention
  layer. With `temporal_window=3` and `n_temporal_layers=1` the overhead is
  modest; memory is dominated by the `(T·V) × (W·V)` logits per joint.

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Temporal attention over-smooths motion. | Start with `temporal_loss_weight=0.1` and `temporal_window=3`. |
| Extra memory from spatio-temporal logits. | Default `temporal_window=3`; can be reduced to `1` to test. |
| Boundary effects for short clips. | Zero-padding plus temporal mask; tests include `T < temporal_window`. |
| Geometry bias computed per-frame is reused across time. | Cameras are static in the datasets; for moving cameras, compute bias per `(t, t+dt)` pair. |

## 10. Future work

* **Temporal ray-intersection bias**: compute ray-intersection quality between
  frames using temporal correspondences, rather than reusing the per-frame
  view-pair bias.
* **Depth-proposal temporal consistency**: regularise the v25 depth-proposal
  head so that depth hypotheses are coherent across neighbouring frames.
* **Learned temporal offsets**: replace the fixed `[-1, 0, +1]` window with
  learned continuous offsets sampled by interpolation.
* **Multi-scale temporal windows**: use different temporal granularities in
  separate heads (short + long).

## 11. References

* `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`
* `motionflow_mv/fusion/temporal_geometry_fusion_v26.py`
* `docs/proposals/v25_multiview_geometry_fusion.md`
