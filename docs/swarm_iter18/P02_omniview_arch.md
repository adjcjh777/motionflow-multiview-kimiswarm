# P02 OmniMultiViewFusion v2 Architecture Design

**Branch:** `feat/swarm-iter18-omniview`  
**Author:** Kimi Code subagent  
**Date:** 2026-08-07  
**Status:** Architecture + skeleton implementation  

## 1. Goal

Design a single, publication-ready multi-view fusion backbone for MotionFlow-MultiView that unifies four previously isolated ideas:

1. **Visibility gating** — explicit per-view/per-joint occlusion reasoning.
2. **Graph-joint attention** — anatomically constrained message passing across views and joints.
3. **Uncertainty-weighted triangulation** — learned anisotropic image-space covariance + adaptive Gauss-Newton refinement.
4. **Spatiotemporal transformer** — joint attention over time and views.

The target is ICRA/CVPR 2027 publishable quality: clean MPJPE **≤ 8.35 mm** on MPI-INF-3DHP S2/Seq1, with stronger robustness under occlusion and calibration noise than the current Bayesian tri v2 ensemble.

## 2. Related work in this repo

| Module | File | What it adds |
|--------|------|--------------|
| `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` | `ray_attention_temporal_crossview_residual_principal_point_model.py` | Principal-point/focal correction + T×V attention + residual MLP. Baseline ~9.32 mm. |
| `RayAttentionFusionModelBayesianTriV2` | `ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py` | Anisotropic 2-D covariance, adaptive GN damping, epipolar loss. |
| `CrossViewGraphAttention` | `fusion/prototypes/cross_view_graph_attention.py` | Sparse (view, joint) graph attention with bone/symmetry/cross-view edges. |
| `VisibilityGatedFusionModel` | `visibility_gated_fusion.py` | Explicit visibility head with fallback guard. |
| `RayAttentionFusionModelSpatiotemporal` | `ray_attention_spatiotemporal_model.py` | Unified T×V×J attention grid. |

None of these combine all four ideas. OmniMultiViewFusion v2 is the first integrated prototype.

## 3. Architecture overview

```
Input: (B, T, V, J, 3)  -> (x, y, confidence)
        |
        v
┌─────────────────────────────────────┐
│  Principal-point / focal correction │  (reuses PrincipalPointCorrection)
│  K -> K_corr, pp_delta, focal_scale   │
└─────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────┐
│  Per-frame ray-aware encoder        │  (obs + ray + camera embed)
│  Output: (B*T, V, J, d)              │
└─────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────┐
│  Visibility head                    │
│  m_vj = sigmoid(MLP(feat))          │
│  Fallback guard for <2 visible views│
└─────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────┐
│  Graph-joint attention block        │
│  Sparse (view, joint) graph with  │
│  bone / symmetry / cross-view edges│
└─────────────────────────────────────┘
        |
        v
─────────────────────────────────────┐
│  Spatiotemporal transformer (T×V)   │
│  Adds time + view positional embed  │
│  Runs over (B*J, T*V, d) tokens    │
└─────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────┐
│  Uncertainty head                   │
│  predicts 2x2 Cholesky factor L_vj  │
│  precision = 1/sqrt(det(Σ))         │
└─────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────┐
│  Weighted DLT + adaptive GN         │
│  weights = w_vj * conf * m_vj * prec│
└─────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────┐
│  Residual refinement MLP            │
│  X_final = X_gn + MLP([feat, X_gn]) │
└─────────────────────────────────────┘
        |
        v
Output: pred_3d, weights, visibility, covariance, epipolar_loss
```

## 4. Detailed component design

### 4.1 Input and camera correction

* Same interface as existing ray-attention models: `x` of shape `(B, T, V, J, 3)`, plus `cameras` or `(K, R, t)`.
* Reuse `PrincipalPointCorrection` to predict `pp_delta` and optional `focal_scale`. The corrected intrinsics `K_corr` are used for ray embedding and triangulation.

### 4.2 Per-frame ray-aware encoder

* Observation embedding: `Linear(3, d/2)`.
* Ray embedding: back-project 2-D points to unit rays using `K_corr, R, t`; embed ray origin + direction via `Linear(6, d/2)`.
* Camera embedding: flatten `K, R, t` and project to `d`.
* Output: `(B*T, V, J, d)`.

### 4.3 Visibility gating

```python
visibility_logits = visibility_head(feat)   # (B*T, V, J)
visibility = sigmoid(visibility_logits)
# fallback guard: if visible count for a joint < min_visible_views, force 1.0
```

* Used as a multiplicative mask in triangulation weights.
* Auxiliary `BCEWithLogitsLoss` against synthetic occlusion labels (or weak labels from confidence).
* Also applied as an attention bias in the graph-joint layer: occluded nodes get reduced edge weights via learned edge-type bias.

### 4.4 Graph-joint attention

* Replace the dense joint self-attention with `CrossViewGraphAttention`.
* Graph nodes: `(v, j)` for each view and joint.
* Edge types:
  * `0` — bone (parent ↔ child) within each view.
  * `1` — symmetry (left/right mirror pairs) within each view.
  * `2` — cross-view same-joint across views.
  * `3` — self-loop.
* Multi-head attention with per-edge-type embeddings and scalar biases.
* Output shape preserved: `(B*T, V, J, d)`.

### 4.5 Spatiotemporal transformer

* Reshape features to `(B, T, V, J, d)`.
* Add learned positional embeddings for time and view.
* Flatten to `(B*J, T*V, d)` tokens and run standard transformer encoder layers.
* This is the same factorized T×V block used by the current cross-view temporal models; it is kept because a full T×V×J grid is too heavy for a single RTX 4090 smoke run.

### 4.6 Uncertainty-weighted triangulation

* A covariance head predicts 3 raw parameters per `(view, joint)`.
* Convert to lower-triangular Cholesky factor `L` (2×2).
* Precision weight: `precision = 1 / (L_xx * L_yy)`.
* Final weight:
  ```
  weights = sigmoid(w_head(feat)) * confidence * visibility * precision
  ```
* Triangulate via fully batched `triangulate_dlt_batched_lstsq`.
* Adaptive Gauss-Newton refinement with per-joint damping predicted from pooled features (Bayesian tri v2 design).

### 4.7 Residual refinement and outputs

* Pool features across views: `feat_pooled = mean(feat, dim=1)`.
* Concatenate with refined 3-D estimate; residual MLP predicts `ΔX`.
* Final pose: `X = X_gn + ΔX`.
* Return tuple:
  * `pred_3d`: `(B, T, J, 3)`
  * `weights`: `(B, T, V, J)`
  * `visibility`: `(B, T, V, J)`
  * `covariance`: `(B, T, V, J, 2, 2)`
  * `epipolar_loss`: scalar auxiliary loss

## 5. Losses and training recipe

| Loss | Weight | Notes |
|------|--------|-------|
| 3D MPJPE | 1.0 | Main regression loss. |
| Visibility BCE | 0.1 | Supervised by synthetic occlusion masks. |
| Epipolar consistency | 0.05 | Weighted by predicted covariance. |
| Bone-length consistency | 0.05 | Keeps skeleton shape. |
| Velocity smoothness | 0.02 | Temporal regularizer. |

**Warm-start strategy:**
1. Load `ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_v2.pth`.
2. Freeze per-frame encoder + T×V transformer for 5 epochs.
3. Train new heads (visibility, graph, uncertainty) only.
4. Unfreeze all; train end-to-end for 15–20 epochs.

## 6. Expected improvements

| Scenario | Current best | OmniMultiView v2 target |
|----------|--------------|-------------------------|
| Clean MPI-INF-3DHP S2/Seq1 | 8.35 mm (ensemble) | ≤ 8.0 mm single model |
| Occlusion 30% | ~12 mm | ≤ 10 mm |
| Calibration pp error 10 px | ~15 mm | ≤ 11 mm |
| Variable views k=2 | ~25 mm | ≤ 18 mm |

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Graph attention adds memory | Use sparse scatter softmax; limit to 1–2 layers. |
| Visibility head degenerates | Fallback guard; supervised synthetic occlusion labels. |
| Uncertainty head explodes | Clamp log-variance to [-5, 5]. |
| Negative interaction with strong baseline | Warm-start + staged unfreezing. |
| Variable view count | Rebuild `edge_index` for the active view subset at runtime. |

## 8. Files

* `docs/swarm_iter18/P02_omniview_arch.md` — this document.
* `motionflow_mv/fusion/omniview_fusion_v2.py` — skeleton implementation with smoke test.

## 9. Next steps

1. Run CPU smoke test: shape/gradient sanity + single-frame compatibility.
2. Queue a small GPU smoke run on MPI-INF-3DHP S1 with `d=48`, 1 graph layer, 10 epochs.
3. Compare against the `bayesian_tri_v2` anchor.
4. If clean MPJPE is within 5% of anchor, scale to full 20-epoch run.
