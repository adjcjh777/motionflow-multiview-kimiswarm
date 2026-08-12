# v53 Multi-View Temporal Synchronization Refinement (MVTS-R)

**Author:** design-swarm  
**Status:** proposal  
**Depends on:** v52-UWT, v46-SVG, v45-AGF  
**Tracking issue:** (to be assigned)

## 1. Motivation

The v52 Uncertainty-Weighted Triangulation (UWT) block learns per-view, per-joint precision weights and re-triangulates on a *per-frame* basis. It therefore assumes that frame `t` in every view captures the same physical instant. Real multi-camera rigs rarely satisfy this: rolling shutter, clock skew, dropped frames, and independent encoders can introduce sub-frame or even frame-level temporal drift across views. When a wrist or ankle moves quickly, a one-frame shift between cameras projects the same 3-D point to very different image locations and degrades triangulation.

**v53 Multi-View Temporal Synchronization Refinement (MVTS-R)** builds directly on the v52 UWT foundation. It takes v52's re-triangulated 3-D pose `X^U` and its learned precision weights `w^U`, predicts a small per-view temporal offset, warps the 3-D trajectory in time for each view, and fuses the warped trajectories using the v52 weights. The module is residual and identity-initialized, so at startup it leaves the v52 output unchanged — a warm-start friendly stack on top of v52.

## 2. Architecture

Insert `multi_view_temporal_sync_v53` **after** `UncertaintyWeightedTriangulationV52` in `OmniMultiViewFusionV5` and **before** the residual MLP / v47/v49 temporal aggregation.

```text
features, points_2d, K, R, t
         |
         v
[ v52-UWT ]  ->  X^U, w^U, loss_uwt
         |
         v
[ v53-MVTS-R ] ->  X^out, loss_mvts, tau_v
```

### 2.1 Temporal-offset predictor

For each view `v` and time `t` compute the per-joint reprojection residual of the v52 estimate:

```
r_{v,t,j} = || π_v( X^U_{t,j} ) - x_{v,t,j} ||_2        # (B, T, V, J)
```

Pool the v52 weight and residual into a per-view, per-time feature:

```
e_{v,t} = concat(
    mean_j(r_{v,t,j}), log(mean_j(r_{v,t,j}) + ε),
    mean_j(w^U_{v,t,j}), std_j(w^U_{v,t,j})
)                                                          # (B, T, V, 4)
```

A small MLP predicts the sub-frame temporal offset per view:

```
τ_{v,t} = max_shift * tanh( MLP_τ( e_{v,t} ) )            # (B, T, V)
```

`max_shift = 2.0` frames by default; `tanh` keeps the offset bounded.

### 2.2 Temporal warping and fusion

Warp the v52 3-D pose sequence for each view using linear interpolation along `T`:

```
X^warp_{v,t,j} = LinearInterp( X^U_{:,j}, t + τ_{v,t} )    # (B, T, V, J, 3)
```

Fuse the warped poses with the v52 precision weights:

```
w̃_{v,t,j} = w^U_{v,t,j} / (Σ_u w^U_{u,t,j})                # normalise over views
X^sync_t = Σ_v w̃_{v,t} * X^warp_{v,t}                        # (B, T, J, 3)
```

The final output is a gated residual so the module can be faded in:

```
X^out = X^U + g * ( X^sync - X^U ),   g = sigmoid(γ), γ initialized to 0
```

### 2.3 Auxiliary loss

Three terms regularize the offsets and encourage cross-view agreement:

1. **Sync consistency:** penalises the distance between each warped view and the fused pose, weighted by the v52 precision:

```
L_sync = Σ_{b,t,v,j} w̃_{v,t,j} || X^warp_{v,t,j} - X^sync_{t,j} ||^2
```

2. **Offset regularization:** keeps offsets small when views are already aligned:

```
L_off = (1 / BTV) Σ_{b,t,v} τ_{v,t}^2
```

3. **Temporal smoothness:** discourages jitter in the predicted offsets:

```
L_smooth = Σ_{b,t,v} (τ_{v,t} - τ_{v,t-1})^2
```

Total auxiliary loss:

```
loss_mvts = λ_sync * L_sync + λ_off * L_off + λ_smooth * L_smooth
```

with defaults `λ_sync=0.01`, `λ_off=0.001`, `λ_smooth=0.001`.

## 3. Inputs and Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_uwt` | `(B, T, J, 3)` | v52 re-triangulated 3-D pose |
| `uwt_weights` | `(B, T, V, J)` | v52 precision weights |
| `points_2d` | `(B, T, V, J, 2)` | Input 2-D keypoints |
| `K`, `R`, `t` | `(B,T,V,3,3)`, `(B,T,V,3)` | Calibrated cameras |
| `view_mask` | `(B, T, V)` | Valid view mask |

**Outputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Temporally synchronized 3-D pose |
| `mvts_loss` | scalar | Auxiliary sync/offset/smoothness loss |
| `offsets` | `(B, T, V)` | Predicted per-view temporal offsets (diagnostic) |

## 4. Config Flags

```yaml
use_v53_multi_view_temporal_sync: false
v53_mvts_hidden: 64
v53_mvts_n_layers: 2
v53_mvts_max_shift: 2.0
v53_mvts_offset_reg_weight: 0.001
v53_mvts_sync_loss_weight: 0.01
v53_mvts_temporal_smoothness_weight: 0.001
v53_mvts_identity_gate_init: 0.0
v53_mvts_use_uwt_weights: true
v53_mvts_use_feature_bias: true
```

## 5. Expected MPJPE Impact

- **Full-view WebBridge/H36M:** 0.5–1.0 mm drop over the v52 baseline, e.g. 17.0 mm → 16.2–16.5 mm.
- **Sparse-view MPJPE@2/3:** 1.5–3.0 mm improvement on fast-motion clips where temporal drift is most harmful.
- **3DPW actual mode:** modest consistent gain by better aligning in-the-wild unsynchronized views.

## 6. Risks

See `docs/swarm_iter27/reports/agent_multi_view_temporal_sync_v53_risks.md`.

## 7. 5-Step Implementation Plan

1. **Module stub:** create `motionflow_mv/fusion/multi_view_temporal_sync_v53.py` implementing `MultiViewTemporalSyncV53` with an offset MLP, a differentiable temporal warp helper, and zero-initialized residual gate.
2. **Integrate into `OmniMultiViewFusionV5`:** add the flags above, instantiate the module after `uncertainty_weighted_triangulation_v52`, and pass `pred_3d_gn_uwt`, `uwt_weights`, `points_2d`, `K`, `R`, `t`, and `view_mask`.
3. **Add smoke config and script:** create `configs/benchmark_v53_mvts_smoke.yaml` and `scripts/run_v53_mvts_smoke_local_4090.sh` that enable v53 on top of the v52 baseline.
4. **Smoke verification on RTX 4090:** confirm identity-at-init (`|MPJPE_v53 - MPJPE_v52| < 0.1 mm`), finite offsets, and stable training without NaN/Inf/OOM.
5. **Full A800 run and ablation:** queue a full run, compare `val_MPJPE` and `MPJPE@2/3/4` against the v52 baseline, and optionally ablate `use_uwt_weights` and `max_shift`.
