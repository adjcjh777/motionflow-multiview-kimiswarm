# v54 Multi-Scale Geometry Fusion (MS-GF v54)

**Author:** design-swarm
**Status:** proposal
**Depends on:** v45-AGF, v46-SVG, v51-CDSVR, v52-UWT, v53-PSC
**Tracking issue:** #186

## 1. Motivation

v52 learns per-view/joint triangulation weights and v53 calibrates the resulting pose against the floor and a canonical skeleton. Both operate on the *final* 3-D point and treat each joint independently, yet real errors are correlated across joints: one bad view shifts a whole limb, and body-scale drift affects every joint together.

**v54 Multi-Scale Geometry Fusion (MS-GF v54)** fuses geometric evidence across four scales—joint, limb, body, and scene/floor—using the v52 weights and v53 physical estimates as guidance. It is warm-start/identity-at-init, so it can be dropped on top of a trained v53 checkpoint.

## 2. Architecture

MS-GF v54 is inserted **after** `PhysicalSpaceCalibrationV53` in `OmniMultiViewFusionV5`.

```text
features, points_2d, K, R, t, pred_3d_v53, uwt_weights, floor_height, bone_scale
                                   |
                                   v
              [ v54-MS-GF ]  ->  pred_3d_v54, w^{out}_{v,j}, loss_msgf
```

Inside the module:

1. **Scale-token builders** construct joint, limb, body, and scene/floor tokens from the v53 pose, v52 weights, and camera geometry.
2. **Hierarchical cross-view attention** runs per scale, masked by `view_mask` and weighted by v52 precision weights.
3. **Cross-scale fusion** updates joint tokens with coarse limb/body/scene tokens via a gated attention mechanism.
4. **Output heads** produce a small 3-D residual `ΔX` and a refined weight multiplier `g_{v,j}`; both are zero-initialized, so the module is an identity at startup.

## 3. Equations

Given the v53-calibrated pose `X ∈ R^{B×T×J×3}`, v52 weights `w^in ∈ R^{B×T×V×J}`, and camera parameters, the per-view per-joint residual is:

```
r_{v,j} = ||π_v(X_j) - x_{v,j}||_2        ∈ R^{B×T×V×J}
```

Joint-scale token (`d_z = d + 5`):

```
z^J_{v,j} = concat( f_{v,j},  r_{v,j},  log(r_{v,j}+ε),  w^in_{v,j},  n_v )
```

where `n_v` is the normalized ray direction for view `v` and joint `j`.

Limb, body, and scene tokens are built by weighted pooling:

```
z^L_{v,l} = MLP_L( Σ_j A_lj · w^in_{v,j} · z^J_{v,j} / Σ_j A_lj · w^in_{v,j} )  ∈ R^h
z^B_v     = MLP_B( Σ_j w^in_{v,j} · z^J_{v,j} / Σ_j w^in_{v,j} )                  ∈ R^h
z^S_v     = MLP_S( concat( z^B_v, floor_height_v, bone_scale_v ) )                ∈ R^h
```

`A_lj` is a learnable soft limb assignment initialized from the dataset skeleton; `floor_height_v` and `bone_scale_v` come from v53.

Per-scale cross-view attention (joint scale):

```
^J_{v,j} = z^J_{v,j} + MHSA_{views}( q=K=Q=z^J_{:,j}, mask=view_mask, bias=γ_J · b^geo )_v
```

`b^geo` combines ray-intersection affinity and epipolar distance from the v25/v45 geometry modules.

Cross-scale update:

```
h^J_{v,j} = ^J_{v,j} + Σ_{s ∈ {L,B,S}} α_{s→J} · g_s( ^s_v )
```

where `g_s` are small MLPs and `α` is a learned gate initialized to zero.

Refined weight and pose correction:

```
g_{v,j} = MLP_g( h^J_{v,j} ),            g_{v,j} = 0  at init
w^out_{v,j} = w^in_{v,j} · (1 + tanh(g_{v,j}))

ΔX_j = MLP_x( pool_v h^J_{v,j} ),        ΔX_j = 0  at init
pred_3d_v54 = pred_3d_v53 + λ · ΔX_j
```

`λ` is a learned scalar gate initialized to `0.0`. The `tanh` keeps the weight multiplier in `[0, 2]`.

Auxiliary loss:

```
L_msgf = v54_msgf_loss_weight · ( mean(r_{v,j} · w^out_{v,j}) + λ_cons · KL(w^out || w^in) )
```

the KL term discourages the refined weights from drifting far from the v52 weights.

## 4. Inputs and Outputs

**Inputs:**
- `features`: `(B, T, V, J, d)`
- `points_2d`: `(B, T, V, J, 2)`
- `K`, `R`, `t`: camera parameters, `(B, T, V, 3, 3)` / `(B, T, V, 3)`
- `pred_3d_v53`: `(B, T, J, 3)`
- `uwt_weights`: `(B, T, V, J)` from v52
- `floor_height`: `(B, T)` from v53
- `bone_scale`: `(B, T, N_bones)` from v53
- `view_mask`: `(B, T, V)`

**Outputs:**
- `pred_3d`: `(B, T, J, 3)`
- `msgf_loss`: scalar auxiliary loss
- `w_out`: `(B, T, V, J)` refined weights
- `scale_attention`: `(B, T, V, V, J)` optional per-scale attention map for diagnostics

## 5. Config Flags

```yaml
use_v54_multi_scale_geometry_fusion: false
v54_msgf_hidden: 64
v54_msgf_n_heads: 4
v54_msgf_n_layers: 2
v54_msgf_scales: ["joint", "limb", "body", "scene"]
v54_msgf_limb_grouping: "h36m_17_limbs"   # or "mpi_28_limbs", "universal_16"
v54_msgf_weight_type: "per_view_joint"    # per_view, per_joint, per_view_joint
v54_msgf_identity_init: true
v54_msgf_residual_gate_init: 0.0
v54_msgf_loss_weight: 0.01
v54_msgf_kl_weight: 0.001
v54_msgf_use_v52_weights: true
v54_msgf_use_v53_floor: true
v54_msgf_use_v53_bone: true
v54_msgf_use_geometry_bias: true
```

## 6. Expected MPJPE Impact

- **Full-view WebBridge/H36M:** 0.5–1.0 mm drop over the v53 baseline (e.g. 16.5 → 15.5–16.0 mm).
- **Sparse-view MPJPE@2:** 2–4 mm improvement from limb/body constraints.
- **Sparse-view MPJPE@3:** 1–2 mm improvement.
- **Cross-domain (3DPW actual):** modest gain from scene-level floor anchoring.

## 7. Risks and Mitigations

See `docs/swarm_iter28/reports/agent_multi_scale_geometry_fusion_v54_risks.md`.

## 8. Implementation Plan

1. **Implement `MultiScaleGeometryFusionV54`** in `motionflow_mv/fusion/multi_scale_geometry_fusion_v54.py`: token builders, per-scale geometry-biased cross-view attention, cross-scale fusion, zero-initialized output layers, and the auxiliary loss.
2. **Wire into `OmniMultiViewFusionV5`**: add flags, instantiate after `physical_space_calibration_v53`, pass `uwt_weights`, `floor_height`, and `bone_scale`, and aggregate `v54_msgf_loss_weight * msgf_loss` into the geometry loss.
3. **Add smoke config and script**: `configs/benchmark_v54_msgf_smoke.yaml` and `scripts/run_v54_msgf_smoke_local_4090.sh`, starting from a v53 checkpoint with `v54_msgf_identity_init=true`.
4. **Smoke test on RTX 4090**: verify identity-at-init (`|MPJPE_v54 - MPJPE_v53| < 0.1 mm`), check `MPJPE@2/3/4`, and ensure no NaN/Inf/OOM.
5. **A800 full run and ablation**: run full training, ablate individual scales (`scene`, `body`, `limb`), update `AGENTS.md`, and report final val_MPJPE.
