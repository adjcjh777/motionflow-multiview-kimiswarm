# v53: SMPL Human-Model Bridge on Uncertainty-Weighted Triangulation

**Author:** design-swarm agent  
**Module name:** `smpl_human_model_bridge_v53`  
**Status:** Proposal (design-only)  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v25, v45, v52 uncertainty-weighted triangulation, v50, v51

## 1. Motivation

v52 Uncertainty-Weighted Triangulation weights per-view evidence but still treats joints independently.  The paper pipeline

> multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline

needs an explicit anthropometric bridge.  v53 converts the v52 skeleton into a SMPL body and blends SMPL-aligned joints back. It is **warm-startable / identity at init**: blend gate starts closed and the final residual is zero-initialized, so a v53-enabled v52 checkpoint returns the same pose.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

```
[v25/v45/v46/v51 fusion]
        |
        v
UncertaintyWeightedTriangulationV52  ->  pred_3d_uwt  (B, T, J, 3)
        |
        v
SMPLHumanModelBridgeV53  ->  pred_3d_smpl  (B, T, J, 3)
        |
        v
[v28/v40 physical alignment, v50 SEFH, losses]
```

### 2.2 Inputs / outputs

Inputs: `pred_3d_uwt (B,T,J,3)`, `weights (B,T,V,J)`, `features (B,T,V,J,d)`, `points_2d (B,T,V,J,2)`, `K,R,t`, `view_mask (B,T,V)`, `domain_id (B,)`.  Output: `pred_3d_smpl (B,T,J,3)`, `smpl_loss`.

### 2.3 Uncertainty-weighted feature pooling

```text
w~_vjt = softmax_v( log W_vjt / τ_uwt ),  τ_uwt = 1.0
f_jt  = Σ_v w~_vjt · feat_vjt               # (B, T, J, d)
```

### 2.4 SMPL parameter head

A 2-layer MLP predicts residual SMPL parameters from `z = concat(pred_3d_uwt, f)`:

```text
Δbody_pose   ∈ R^(B,T,69)        # 23 SMPL joints × 3
Δbetas       ∈ R^(B,T,10)        # temporally pooled shape
global_rot   ∈ R^(B,T,3)
transl       ∈ R^(B,T,3)
```

At init, final layers are zero, so the pose is canonical and `global_rot = transl = betas = 0` (mean shape).

### 2.5 SMPL forward and joint regression

```text
V_smpl, J_smpl = SMPL(body_pose + Δbody_pose,
                       betas + Δbetas,
                       global_rot + Δglobal_rot,
                       transl + Δtransl)
# V_smpl: (B, T, 6890, 3), J_smpl: (B, T, 24, 3)

P_smpl = J_smpl · M^T           # (B, T, J, 3), M ∈ R^(J×24)
```

If an SMPL file is unavailable, a learned fallback regressor predicts `P_smpl` directly.

### 2.6 Identity-at-init blending

```text
α      = sigmoid(g(f))          # (B, T, J, 1), sigmoid(-5) ≈ 0.007 at init
P_out  = P_uwt + α · (P_smpl - P_uwt)
ΔP     = v53_smpl_residual_gate · MLP_residual(P_out - P_uwt)   # zero init
P_out  = P_uwt + ΔP
```

With `v53_smpl_residual_gate = 0.0` and gate init `-5.0`, the module is exactly identity.

### 2.7 Auxiliary losses

```text
L_smpl = λ_3d |P_smpl - P_gt|_2
       + λ_reproj Σ_v W_v ||π_v(P_smpl) - x_v||_2^2
       + λ_bone Σ_bone (||bone|| - μ_bone)^2
       + λ_floor max(0, -min_y V_smpl)
       + λ_inter Σ_{i<j} max(0, r - ||v_i - v_j||)^2
```

## 3. Configuration flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v53_smpl_human_model_bridge` | bool | `False` | Enable the module. |
| `v53_smpl_hidden` | int | `64` | Parameter MLP hidden dim. |
| `v53_smpl_shape_dim` | int | `10` | SMPL shape dimension. |
| `v53_smpl_loss_weight` | float | `0.01` | Multiplier on `L_smpl`. |
| `v53_smpl_max_gate` | float | `0.8` | Maximum blend gate value. |
| `v53_smpl_reproj_weight` | float | `0.1` | Reprojection loss weight. |
| `v53_smpl_bone_weight` | float | `0.1` | Bone-length prior weight. |
| `v53_smpl_floor_weight` | float | `0.05` | Floor-penetration weight. |
| `v53_smpl_intersection_weight` | float | `0.01` | Self-intersection proxy weight. |
| `v53_smpl_warmup_epochs` | int | `0` | Epochs before applying `L_smpl`. |
| `v53_smpl_identity_init` | bool | `True` | Zero-init outputs and close gate. |
| `v53_smpl_fast_mode` | bool | `False` | Skip the 6890-vertex SMPL mesh. |
| `v53_smpl_blend_gate_init` | float | `-5.0` | Initial gate logit. |
| `v53_smpl_residual_gate` | float | `0.0` | Scale of zero-initialized residual. |
| `v53_smpl_model_path` | str | `None` | Path to `SMPL_NEUTRAL.pkl`. |

## 4. Expected MPJPE impact

| Scenario | Expected delta |
|---|---|
| Full-view H36M / MPI-INF-3DHP | −1 to −2.5 mm |
| Sparse 2–3 view (v46/v51) | −2 to −4 mm on `MPJPE@2/3` |
| WebBridge / 3DPW actual mode | −1.5 to −3.5 mm on extreme poses |
| Combined with v50 SEFH | up to −3 to −5 mm on `MPJPE@full` |



## 5. Risks

See `docs/swarm_iter27/reports/agent_smpl_human_model_bridge_risks.md`.

## 6. 5-step implementation plan

1. **Module stub.** Create `motionflow_mv/fusion/smpl_human_model_bridge_v53.py` with uncertainty-weighted pooling, SMPL parameter MLPs, optional `smplx` forward, a fallback skeleton regressor, the identity blend, and gated residual.  Zero-initialize all final output layers.

2. **Wiring in `OmniMultiViewFusionV5`.** Instantiate the module after the v52 UWT block and call it with `pred_3d_uwt`, `features`, `weights`, cameras, `view_mask`, and `domain_id`.  Feed the result into v28/v40 physical alignment and v50.

3. **Loss plumbing.** Add `L_smpl` to the trainer’s auxiliary loss dictionary, gated by `v53_smpl_warmup_epochs`.  Expose SMPL residuals to the v50 Self-Evolution Feedback Head.

4. **Smoke and warm-start tests.** Create `configs/benchmark_v53_smpl_bridge_smoke.yaml` and `scripts/run_v53_smpl_bridge_smoke_local_4090.sh`.  Load a v52 checkpoint, enable v53 with `v53_smpl_residual_gate=0.0`, and assert `val_MPJPE` changes by less than `0.1 mm`.

5. **Scale to A800.** Add an A800 queue entry in `scripts/launch_v33_a800_queue.py` on top of the strongest v52 checkpoint and report `val_MPJPE`, `MPJPE@k`, and SMPL samples.
