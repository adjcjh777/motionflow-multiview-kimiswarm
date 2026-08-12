# v52 — Physical-Space Calibration (PSC)

## Motivation

The MotionFlow-MultiView pipeline is:

```
multi-view video → 2D pose → multi-view fusion/triangulation →
camera calibration → physical-space alignment → motionflow output
```

v25–v51 improved fusion, sparse-view reliability, temporal coherence, and domain generalisation, but **camera calibration is still mostly fixed input**. Small calibration errors propagate into triangulated poses and are only indirectly corrected by downstream pose-refinement modules (v28, v40). v52 closes the loop by making camera calibration a **learnable, physically constrained refinement step** that uses both reprojection residuals and physical priors (floor contact, bone-length constancy, gravity) to jointly update the 3-D pose and the cameras.

## Architecture

`PhysicalSpaceCalibrationV52` is inserted after triangulation and before v28 physical-space alignment. It is identity-at-init.

### Core blocks

1. **Physical feature encoder**  
   Input: `pred_3d ∈ R^(B×T×J×3)`  
   Output: `f_phys ∈ R^(B×T×F)` where `F = 3J + n_bones + 4`  
   Extracts root-centred joint positions, bone vectors/lengths, floor height `h_floor = min_y(pred_3d)`, root velocity, and gravity projection statistics.

2. **Camera refinement head**  
   Input: `f_phys` and per-view mean reprojection residual `e_reproj ∈ R^(B×T×V)`.  
   Outputs (bounded, identity at init):
   - `Δaxis ∈ R^(B×T×V×3)` bounded to `±max_rotation_deg`, applied via SO(3) exponential.
   - `Δt ∈ R^(B×T×V×3)` bounded to `±max_translation_m`.
   - Optional `Δlog_fx, Δlog_fy, Δcx, Δcy` if `v52_psc_refine_intrinsics=True`.

   Refinement is gated:
   ```
   R' = so3_exp(λ · Δaxis) · R
   t' = t + λ · Δt
   K' = K + λ · ΔK            (if enabled)
   λ  = tanh(gate),  gate init = 0
   ```

3. **Pose refinement head**  
   Input: `f_phys` plus camera-corrected reprojection residual.  
   Output: `ΔX ∈ R^(B×T×J×3)` with `|ΔX| ≤ v52_psc_pose_residual_max_m`.  
   The final layer is zero-initialised so `ΔX = 0` at startup.

4. **Physical consistency loss**  
   Added to the training objective with configurable warmup:
   - **Reprojection consistency**: weighted reprojection error under refined cameras.
   - **Bone-length constancy**: MSE between bone lengths of refined pose and a per-sequence moving average.
   - **Floor contact**: hinge penalty on foot joints below the estimated floor plane.

### Equations

```
 f_phys  = Encoder(X)
 Δcam    = CameraHead(f_phys, e_reproj(x, X, K, R, t))
 ΔX      = PoseHead(f_phys)
 (K', R', t') = UpdateCameras((K, R, t), Δcam, gate)
 X'           = X + gate · ΔX

 L_psc   = v52_psc_reproj_weight · L_reproj(x, X', K', R', t'; w)
         + v52_psc_bone_weight   · L_bone(X')
         + v52_psc_floor_weight  · L_floor(X')
```

## Inputs and outputs

- **Inputs**: `pred_3d (B,T,J,3)`, `feat (B,T,V,J,C)`, `K (B,T,V,3,3)`, `R (B,T,V,3,3)`, `t (B,T,V,3)`, `points_2d (B,T,V,J,2)`, `confidences (B,T,V,J)`, `view_mask (B,T,V)`.
- **Outputs**: `pred_3d_refined (B,T,J,3)`, `K_refined`, `R_refined`, `t_refined`, `psc_loss` scalar.

## Integration into `OmniMultiViewFusionV5`

```python
self.use_v52_physical_space_calibration = use_v52_physical_space_calibration
if self.use_v52_physical_space_calibration:
    from motionflow_mv.fusion.physical_space_calibration_v52 import PhysicalSpaceCalibrationV52
    self.physical_space_calibration_v52 = PhysicalSpaceCalibrationV52(
        n_views=self.n_views, n_joints=self.j,
        hidden=v52_psc_hidden, n_layers=v52_psc_n_layers,
        refine_intrinsics=v52_psc_refine_intrinsics,
        refine_extrinsics=v52_psc_refine_extrinsics,
        max_rotation_deg=v52_psc_max_rotation_deg,
        max_translation=v52_psc_max_translation_m,
        pose_residual_max=v52_psc_pose_residual_max_m,
    )
```

Insert in `forward` **after** triangulation and **before** v28 physical-space alignment, adding `psc_loss * schedule_weight` to `epi_loss`.

## Config flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v52_physical_space_calibration` | bool | `False` | Enable the module. |
| `v52_psc_hidden` | int | `64` | Hidden dim of camera/pose head MLPs. |
| `v52_psc_n_layers` | int | `2` | MLP layers per head. |
| `v52_psc_refine_extrinsics` | bool | `True` | Allow R/t updates. |
| `v52_psc_refine_intrinsics` | bool | `False` | Allow focal/pp updates. |
| `v52_psc_max_rotation_deg` | float | `1.0` | Bound on rotation correction. |
| `v52_psc_max_translation_m` | float | `0.05` | Bound on translation correction. |
| `v52_psc_pose_residual_max_m` | float | `0.05` | Bound on per-joint pose correction. |
| `v52_psc_reproj_weight` | float | `1.0` | Reprojection loss weight. |
| `v52_psc_bone_weight` | float | `0.01` | Bone-length loss weight. |
| `v52_psc_floor_weight` | float | `0.01` | Floor-contact loss weight. |
| `v52_psc_warmup_epochs` | int | `0` | Linear warmup of auxiliary loss. |
| `v52_psc_identity_gate_init` | bool | `True` | Zero-initialise final layers/gates. |

## Expected MPJPE impact

- **Baseline v51**: ~17–18 mm A800-D full eval.
- **v52 target**: **0.5–2.0 mm improvement**, especially on noisy-calibration sequences (e.g., WebBridge outdoor captures).
- Complementary to v28: v52 fixes camera/pose geometry; v28 then enforces temporal bone-length and floor constraints.

## Risks

See `docs/swarm_iter26/reports/agent_physical_space_calibration_risks.md`.

## 5-step implementation plan

1. **Module skeleton** — Create `motionflow_mv/fusion/physical_space_calibration_v52.py` with `PhysicalSpaceCalibrationV52`, `PSCFeatureEncoder`, `PSCCameraHead`, and `PSCPoseHead`; add identity-at-init smoke tests.
2. **Geometry kernels** — Implement `so3_exp`, reprojection residual, bone-length targets, and floor-plane extraction; unit-test on synthetic data.
3. **Loss and schedule** — Add `v52_psc_loss` (bone/floor/reprojection) with per-epoch warmup in the trainer.
4. **OmniMultiViewFusionV5 wiring** — Add the toggle in `__init__`, call the module after triangulation, return refined cameras/pose, and pass the warmup weight from the trainer.
5. **Smoke and ablation** — Run a smoke config on RTX 4090, compare to v51 baseline, and queue the full A800-D run if `val_MPJPE` is within 2 mm of baseline.
