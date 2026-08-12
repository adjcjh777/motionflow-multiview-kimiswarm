# v53 Multi-Scale Geometry Fusion (MS-GF)

**Author:** design-swarm  
**Status:** proposal  
**Depends on:** v45-AGF, v46-SVG, v51-CDSVR, v52-UWT  
**Tracking issue:** #185  

## 1. Motivation

v52 Uncertainty-Weighted Triangulation learns per-view, per-joint precision weights and refines the DLT output with a gated residual. Its prediction is *local*: each `(view, joint)` weight uses only that joint’s own feature and residual. Real errors are correlated across joints through foreshortening, depth ambiguity, and occlusion, so they are best explained at **multiple geometric scales**—joint, limb, and body.

**v53 Multi-Scale Geometry Fusion (MS-GF)** builds on v52 by fusing the v52 weights across views and across joint/limb/body scales to produce refined weights and a small scale-aware 3-D correction. It is warm-start/identity-at-init: zero-initialized final layers pass the v52 result through unchanged, so it can be stacked onto an existing v52 checkpoint.

## 2. Architecture

MS-GF is a lightweight graph-attention block inserted **after** `UncertaintyWeightedTriangulationV52` in `OmniMultiViewFusionV5`.

```text
features, points_2d, K, R, t, pred_3d_init, view_mask
         |
         v
[ v52-UWT ]  ->  pred_3d_v52, w^in_{v,j}, loss_uwt
         |
         v
[ v53-MS-GF ] ->  pred_3d_v53, loss_msgf, refined weights w^out_{v,j}
```

Inside v53:

1. **Scale token builders** form joint/limb/body tokens from residuals, v52 weights, and input features.
2. **Cross-view attention** runs per scale (masked by `view_mask`) to capture per-scale view consistency.
3. **Cross-scale fusion** updates joint tokens from limb and body tokens.
4. **Weight refinement head** predicts a factor `g_{v,j}` from the fused token; the zero-initialized final layer gives `w^out_{v,j} = w^in_{v,j} * (1 + tanh(g_{v,j})) = w^in_{v,j}` at startup.
5. **3-D residual correction head** predicts `ΔX_j` and is also zero-initialized, so `pred_3d_v53 = pred_3d_v52` at startup.

## 3. Equations

Given the current 3-D estimate `X ∈ R^{B×T×J×3}` and camera parameters, compute per-view per-joint reprojection residual:

```
r_{v,j} = ||π_v(X_j) - x_{v,j}||_2        ∈ R^{B×T×V×J}
```

Joint-scale token (`d_z = d + 3`):

```
z^J_{v,j} = concat( f_{v,j},  r_{v,j},  log(r_{v,j}+ε),  w^in_{v,j} )  ∈ R^{d_z}
```

Limb/body tokens are obtained by max-pooling joint tokens over the relevant joint set, then projecting with a shared MLP:

```
z^L_{v,l} = MLP_L( maxpool_{jlimb_l} z^J_{v,j} )  ∈ R^h
z^B_v     = MLP_B( maxpool_j z^J_{v,j} )            ∈ R^h
```

Cross-view attention within each scale uses multi-head self-attention over the view dimension (keys/queries per joint/limb/body). For the joint scale:

```
^J_{v,j} = z^J_{v,j} + MHSA_{views}( z^J_{:,j} )
```

Cross-scale update:

```
h^J_{v,j} = concat( ^J_{v,j},  ẑ^L_{v,limb(j)},  ẑ^B_v )
```

Refined weight and pose correction:

```
g_{v,j} = MLP_g( h^J_{v,j} ),            g_{v,j} = 0  at init
w^out_{v,j} = w^in_{v,j} * (1 + tanh(g_{v,j}))

ΔX_j = MLP_x( pool_v h^J_{v,j} ),        ΔX_j = 0  at init
pred_3d_v53 = pred_3d_v52 + γ * ΔX_j
```

`ε = 1e-6`, `γ` is a learned scalar gate initialized to 0. The tanh keeps the weight multiplier in `[0, 2]`.

## 4. Inputs and Outputs

**Inputs** (same tensor shapes as v52):
- `features`: `(B, T, V, J, d)`
- `points_2d`: `(B, T, V, J, 2)`
- `K`: `(B, T, V, 3, 3)`
- `R`: `(B, T, V, 3, 3)`
- `t`: `(B, T, V, 3)`
- `pred_3d_init`: `(B, T, J, 3)`
- `view_mask`: `(B, T, V)` bool
- `w_in`: `(B, T, V, J)` — the v52 precision weights (or identity if v52 disabled)

**Outputs:**
- `pred_3d`: `(B, T, J, 3)`
- `ms_loss`: scalar auxiliary loss
- `w_out`: `(B, T, V, J)` refined weights
- `scale_attention`: `(B, T, V, V, J)` optional cross-view attention map for diagnostics

## 5. Config Flags

```yaml
use_v53_multi_scale_geometry_fusion: false
v53_msgf_hidden: 64
v53_msgf_n_heads: 4
v53_msgf_n_layers: 2
v53_msgf_limb_grouping: "h36m_17_limbs"   # or "mpi_28_limbs", "universal_16"
v53_msgf_weight_type: "per_view_joint"    # per_view, per_joint, per_view_joint
v53_msgf_identity_init: true
v53_msgf_residual_gate_init: 0.0
v53_msgf_loss_weight: 0.01
v53_msgf_use_v52_weights: true
v53_msgf_use_body_scale: true
v53_msgf_use_limb_scale: true
```

## 6. Expected MPJPE Impact

- **Full-view WebBridge/H36M:** 0.5–1.0 mm drop over the v52 baseline (e.g. 17.0 mm → 16.0–16.5 mm).
- **Sparse-view MPJPE@2:** 2–4 mm improvement, because limb/body scale consistency is most valuable when only two views are available.
- **Sparse-view MPJPE@3:** 1–2 mm improvement.
- **Cross-domain (3DPW actual):** modest but consistent gain by reducing correlated body-scale drift.

## 7. Risks and Mitigations

See `docs/swarm_iter27/reports/agent_multi_scale_geometry_fusion_v53_risks.md`.

## 8. Implementation Plan

1. **Implement `MultiScaleGeometryFusionV53`** in `motionflow_mv/fusion/multi_scale_geometry_fusion_v53.py`: token builders, cross-view attention, cross-scale fusion, zero-initialized output layers, and the auxiliary loss.
2. **Wire into `OmniMultiViewFusionV5`**: add flags, instantiate after `uncertainty_weighted_triangulation_v52`, pass `w_in` from v52, and aggregate `v53_msgf_loss_weight * ms_loss` into the geometry loss.
3. **Add smoke config and script**: `configs/benchmark_v53_msgf_smoke.yaml` and `scripts/run_v53_msgf_smoke_local_4090.sh`, starting from a v52 checkpoint with `v53_msgf_identity_init=true`.
4. **Smoke test on RTX 4090**: verify identity-at-init (`|MPJPE_v53 - MPJPE_v52| < 0.1 mm`), check `MPJPE@2/3/4`, and ensure no NaN/Inf/OOM.
5. **A800 full run and ablation**: run full training, ablate `use_limb_scale` and `use_body_scale`, update `AGENTS.md` status tables, and report final val_MPJPE.
