# v54 Geometry-Aware Attention Pooling (GAAP)

**Module name:** `geometry_aware_attention_pooling_v54`  
**Base:** `OmniMultiViewFusionV5` with `use_v52_uncertainty_weighted_triangulation` and `use_v53_physical_space_calibration` enabled.  
**Tracking issue:** TBD (swarm-iter28 / v54)

---

## 1. Motivation

The current v52/v53 pipeline treats multi-view fusion as a two-step process:
1. v52 predicts scalar precision weights per `(view, joint)` and re-triangulates with weighted DLT.
2. v53 refines the resulting 3-D pose against physical-space invariants.

What is missing is an *explicit geometric attention* step that reasons about the 2-D evidence **before** triangulation. In sparse or partially occluded views, a learned per-view scalar weight cannot express the fact that a joint may be visible in one view but poorly aligned across the epipolar geometry. We therefore propose a `GeometryAwareAttentionPoolingV54` block that sits **between the v25/v45 triangulated 3-D estimate and the v52 UWT module**, replacing the implicit average/max pooling over views with a geometry-biased cross-view attention operation.

This module directly aligns with the project narrative: **multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized MotionFlow pipeline**. GAAP refines the "multi-view fusion" stage by letting the model pool 2-D evidence conditioned on 3-D ray geometry, which in turn feeds a better triangulation target for v52 and a better calibrated pose for v53.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

```
[encoder tokens] (B, T, V, J, d)
        │
        ▼
[MultiViewGeometryFusionV25 / v45 adaptive fusion]
        │
        
[initial pred_3d_init via DLT]  (B, T, J, 3)
        │
        ▼
─────────────────────────────────────────┐
│  v54 Geometry-Aware Attention Pooling │  ← NEW
└─────────────────────────────────────────┘
        │
        
[pooled per-joint geometry features]  (B, T, J, d_pool)
        │
        ▼
[UncertaintyWeightedTriangulationV52]
        │
        ▼
[PhysicalSpaceCalibrationV53]
```

### 2.2 Inputs / Outputs

**Inputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `features` | `(B, T, V, J, d)` | per-view encoder tokens from v25/v45 |
| `points_2d` | `(B, T, V, J, 2)` | detected 2-D keypoints |
| `K` | `(B, T, V, 3, 3)` | intrinsic matrices |
| `R` | `(B, T, V, 3, 3)` | rotation matrices |
| `t` | `(B, T, V, 3)` | translation vectors |
| `pred_3d_init` | `(B, T, J, 3)` | initial triangulated 3-D pose from v25/v45 |
| `view_mask` | `(B, T, V)` bool | valid-view mask |
| `uwt_weights` (optional) | `(B, T, V, J)` | v52 precision weights if v52 is stacked *inside* v54 |

**Outputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pooled_features` | `(B, T, V, J, d)` | refined view/joint tokens, used by v52 as `features` |
| `geometry_attention_map` | `(B, T, V, V, J)` | per-joint cross-view attention scores (for diagnostics/entropy loss) |
| `gaap_aux_loss` | scalar or `None` | optional entropy/ray consistency loss term |

If `v54_gaap_pool_type == "cross_view_joint"`, the module returns tokens with the same shape `(B, T, V, J, d)` so that downstream modules do not need shape changes.

## 3. Equations

### 3.1 Geometry-aware attention scores

For a given joint `j` and time `t`, each view `i` attends over all other views `k`:

```
q_i = W_q  f_i                              ∈ R^d          (query = current view)
k_k = W_k  f_k                              ∈ R^d          (key = target view)
e_geom(i, k) = ray_similarity(x_j, c_i, c_k) ∈ R            (pre-computed geometry bias)

score(i, k) = (q_i · k_k) / sqrt(d)  +  β · e_geom(i, k)   (1)
A(i, k)     = softmax_k( score(i, k) + mask(i, k) )        (2)
```

where `β = v54_gaap_geometry_bias_weight` is a learned or fixed scalar.

### 3.2 Ray-geometry bias

For each joint `j`, compute the ray from camera center through the 2-D detection in view `i` and the ray in view `k`. The bias rewards views whose rays nearly intersect the current 3-D estimate:

```
r_i = R_i^T · K_i^{-1} · [u_i, v_i, 1]^T            (direction of ray i in world)
C_i = -R_i^T · t_i                                     (camera center in world)
d_ik = || (C_i + s_i r_i) - (C_k + s_k r_k) ||         (minimum distance between rays)
b(i, k) = exp( -d_ik / σ )                             (geometry affinity, σ = v54_gaap_ray_sigma)

e_geom(i, k) = log( b(i, k) + ε )                      (stable log bias)
```

In practice, the closest-approach distance is computed once per `(time, joint, view-pair)` using `pred_3d_init` as a cheap anchor, i.e., solving for `s_i, s_k` via the skew-line distance.

### 3.3 Output pooling

```
attn_out(i) = Σ_k A(i, k) · (W_v f_k + γ · ray_embed_k)        (3)
f'_i        = LayerNorm( f_i + Dropout(attn_out(i)) )          (4)
```

where `ray_embed_k` is a small `(d,)` vector derived from the 3-D ray direction and `γ` is a learned gate initialised to zero so that the residual is zero at init.

If `v54_gaap_pool_type == "mean"`, the module degenerates to a geometry-weighted mean over views (useful for ablations).

### 3.4 Identity-at-init property

The module is warm-startable:

* The final linear projection inside the attention output (`W_v` → output `d` dims) is zero-initialised.
* `γ` is initialised to `0.0`, so `f'_i == f_i` before training.
* The geometry bias `β` is initialised to `0.0`, so the very first forward pass uses plain content attention only; geometric bias is learned gradually.

Therefore, loading a v53 checkpoint with v54 enabled should change `val_MPJPE` by less than 0.1 mm.

## 4. Config Flags

Add the following to `OmniMultiViewFusionV5.__init__` and the YAML config:

```yaml
use_v54_geometry_aware_attention_pooling: false
v54_gaap_hidden: 64
v54_gaap_n_heads: 4
v54_gaap_n_layers: 2
v54_gaap_pool_type: "cross_view_joint"    # options: cross_view_joint | mean | max
v54_gaap_geometry_bias_weight: 1.0       # scalar β in Eq. (1)
v54_gaap_ray_sigma: 0.1                  # σ in ray-affinity kernel
v54_gaap_residual_gate_init: 0.0          # γ init, 0.0 = identity
v54_gaap_dropout: 0.1
v54_gaap_use_uwt_weights: true            # use v52 weights as attention mask/value bias
v54_gaap_loss_weight: 0.001              # weight for attention entropy regulariser
v54_gaap_identity_init: true
```

## 5. Expected MPJPE Impact

| Scenario | Expected change | Rationale |
|----------|-----------------|-----------|
| Smoke (d=64, 500 samples) | neutral to -0.5 mm | module is identity at init; 2 epochs is too short for geometric bias to dominate |
| Full A800 (d=128, full data) | -0.4 to -1.2 mm | geometric bias improves pooling when v52 weights are noisy or views are sparse |
| Sparse-view test (`V=2,3`) | larger relative gain | ray-similarity attention compensates for missing/incorrect v52 weights |
| With v46/v51 dropout | -0.6 to -1.0 mm | directly complements variable-view reliability |

Conservative target for the smoke test: `val_MPJPE` within 0.2 mm of the v53 baseline and no NaN/OOM.

## 6. Risks and Mitigations

See companion report: `docs/swarm_iter28/reports/agent_geometry_aware_attention_pooling_risks.md`.

## 7. Implementation Plan

1. **Prototype module** (`motionflow_mv/fusion/geometry_aware_attention_pooling_v54.py`) implementing the equations above with identity init and shape-compatible I/O.
2. **Wire into `omniview_fusion_v5.py`** after the v25/v45 triangulation block and before the v52 UWT module; add the ten config flags and instantiate the module when `use_v54_geometry_aware_attention_pooling=True`.
3. **Add YAML smoke config** `configs/benchmark_v54_gaap_smoke.yaml` that extends `benchmark_v53_physical_space_calibration_smoke.yaml` with the v54 flags and a 2-epoch / 500-sample schedule.
4. **Add smoke script** `scripts/run_v54_gaap_smoke_local_4090.sh` and an A800 queue entry in `scripts/launch_v33_a800_queue.py`.
5. **Verify identity-at-init**: load a v53 checkpoint with v54 enabled, run one validation pass, and confirm `ΔMPJPE < 0.1 mm`; then run smoke and compare against v53.
