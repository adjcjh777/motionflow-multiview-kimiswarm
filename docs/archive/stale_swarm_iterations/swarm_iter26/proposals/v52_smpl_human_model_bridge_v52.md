# v52 SMPL Human-Model Bridge

## Motivation

MotionFlow currently fuses calibrated multi-view 2D keypoints into 3D joints (v25/v45), refines them with attention/graph modules (v33–v39), and aligns them with physical priors (v28/v40).  What is still missing is an explicit bridge between the learned 3D skeleton and a **parametric human body model**.  Adding that bridge makes the pipeline consistent with the paper narrative

> multi-view video → human pose extraction → multi-view fusion/calibration → physical-space alignment → optimized MotionFlow,

because SMPL is the canonical representation that turns sparse 3D joints into a physically plausible body with mesh, bone lengths, and floor contact.  v52 introduces a lightweight, optional **SMPL Human-Model Bridge** that converts the fused 3D pose into SMPL parameters, runs the parametric body, and blends the regressed SMPL joints back into the pipeline as a physical-space refinement.  The module is **warm-startable/identity-at-init**: at initialization the blend weight is zero, so the existing v51 baseline is reproduced exactly.

## Architecture

The bridge is inserted **after** adaptive Gauss–Newton triangulation and **before** v28/v40 physical-space alignment and the v50 Self-Evolution Feedback Head.

### Inputs/outputs

Inputs from `OmniMultiViewFusionV5`:

* `P_fused ∈ R^(B,T,J,3)` — fused 3D joints
* `feat ∈ R^(B,T,V,J,d)` — per-view encoder features
* `view_mask ∈ {0,1}^(B,T,V)` — active views
* `domain_id ∈ Z^B` — optional domain label

Output:

* `P_out ∈ R^(B,T,J,3)` — SMPL-aware 3D joints

### Feature pooling

```text
f_pooled^(b,t,j) = Σ_v mask[b,t,v] · feat[b,t,v,j] / Σ_v mask[b,t,v]   ∈ R^d
z = concat(P_fused, f_pooled)                                            ∈ R^(B,T,J,d+3)
```

### SMPL parameter head

A 2-layer MLP/Transformer consumes `(B·T, J, d+3)` tokens:

* `betas ∈ R^10` — clip-level shape
* `body_pose ∈ R^(B,T,69)` — 23 SMPL joints × 3
* `global_orient ∈ R^(B,T,3)`
* `transl ∈ R^(B,T,3)`

At init, pose heads are biased to the **canonical SMPL pose** and shape to the **mean SMPL shape**, so the body starts neutral.

### SMPL forward and joint regression

If `smplx` is available, `smplx.SMPL` produces

```text
V_smpl ∈ R^(B,T,6890,3)    (vertices)
J_smpl ∈ R^(B,T,24,3)      (SMPL joints)
```

A learned regressor `M ∈ R^(J×24)` maps SMPL joints to the target skeleton:

```text
P_smpl = M · J_smpl   ∈ R^(B,T,J,3)
```

When `smplx` is unavailable, the predicted parameters are mapped through the same regressor, keeping the module testable without the model file.

### Identity-at-init blending

```text
P_out = P_fused + α · (P_smpl - P_fused)
```

`α = sigmoid(g(f_pooled)) ∈ (0,1)^(B,T,J,1)`.  The final projection is **zero-initialized** and `v52_smpl_blend_weight_init = 0.0`, so `α ≈ 0` at start and the baseline is preserved.

### Losses

```text
L_v52 = λ_3d · |P_smpl - P_gt|_2
      + λ_proj · L_reproj(P_smpl, cameras, keypoints_2d)
      + λ_bone · Σ_bone (‖P_smpl[parent]-P_smpl[child]‖ - μ_bone)^2
      + λ_floor · L_floor(V_smpl)
```

## Integration into `OmniMultiViewFusionV5`

The bridge follows the v46–v51 plugin pattern:

```python
if self.use_v52_smpl_human_model_bridge and self.smpl_bridge_v52 is not None:
    pred_3d_gn = self.smpl_bridge_v52(
        pred_3d_gn,
        feat=feat,                 # (B,T,V,J,d)
        view_mask=view_mask_flat,  # (B,T,V)
        domain_id=domain_id,
    )
```

Placed after GN refinement, the call feeds SMPL-aware joints into v28/v40 and v50.

## Config flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v52_smpl_human_model_bridge` | bool | `False` | Enable the module. |
| `v52_smpl_model_path` | str | `None` | Path to `SMPL_NEUTRAL.pkl`. |
| `v52_smpl_hidden` | int | `64` | Hidden dim of the parameter MLP. |
| `v52_smpl_shape_dim` | int | `10` | Number of SMPL shape parameters. |
| `v52_smpl_blend_weight_init` | float | `0.0` | Offset for identity-at-init. |
| `v52_smpl_loss_weight` | float | `1.0` | Overall multiplier on `L_v52`. |
| `v52_smpl_bone_length_weight` | float | `0.1` | Bone-length term weight. |
| `v52_smpl_floor_contact_weight` | float | `0.05` | Floor-contact term weight. |
| `v52_smpl_identity_init` | bool | `True` | Zero-initialize blend projection. |

## Expected MPJPE impact

* **WebBridge / H36M**: ~3–6 mm on occluded/extreme-pose frames.
* **MPI-INF-3DHP / 3DPW**: up to ~8 mm where depth ambiguity is largest.
* **Sparse-view (v46/v51)**: ~2 mm gain by providing an anatomical prior with only 2 views.

## Risks

1. **SMPL dependency / model file**: make the parametric forward optional; keep a fallback learned regressor so the module is testable without `SMPL_NEUTRAL.pkl`.
2. **Training slowdown**: cache the SMPL body per device and offer a fast mode that skips the 6890-vertex mesh forward.
3. **Regressor mismatch / over-smoothing**: initialize `M` with the official SMPL-to-target mapping and clamp `α ≤ 0.8` so the prior cannot override strong fused evidence.
4. **Shape/camera drift**: regularize `‖betas‖²` and constrain the global translation head near the triangulated root.

## 5-step implementation plan

1. **Module stub** (`motionflow_mv/fusion/smpl_human_model_bridge_v52.py`): implement `SMPLHumanModelBridgeV52` with shape/pose heads, optional `smplx` forward, joint regressor, and zero-initialized blend.
2. **Wiring in `OmniMultiViewFusionV5`**: add the config flags, instantiate the module, and insert the forward call after Gauss–Newton refinement.
3. **Loss terms**: add 3D, reprojection, bone-length, and floor-contact losses to the trainer’s auxiliary loss dictionary.
4. **Smoke test**: create `configs/benchmark_v52_smpl_bridge_smoke.yaml` and run a 50-sample smoke on RTX 4090; verify val_MPJPE does not regress vs. v51.
5. **A800 full run**: add an entry to `scripts/launch_v33_a800_queue.py` and measure epoch-1 val_MPJPE against the v51 CDSVR baseline.
