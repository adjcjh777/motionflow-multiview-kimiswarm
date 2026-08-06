# Proposal: Learnable Camera-Centric Coordinate Transform

## One-sentence hypothesis

Replacing the fixed camera-to-world mapping with a *learned, per-view residual SE(3)+scale transform*—conditioned on deep spatio-temporal features and applied after principal-point correction—will better align camera-centric observations to a shared physical world frame, improving multi-view triangulation robustness and lowering clean MPJPE on MPI-INF-3DHP.

## Related existing files / modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — iter14 anchor model (MPJPE 9.32 mm on MPI-INF-3DHP S2/Seq1).
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_module.py` — FusionModule wrapper for the anchor.
- `motionflow_mv/fusion/principal_point_correction.py` — learned intrinsics correction already used by the anchor.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py` — parent class providing `_extract_frame_features` and spatio-temporal transformer.
- `motionflow_mv/fusion/ray_attention_model.py` — `_compute_rays`, `_triangulate_weighted_dlt`.
- `motionflow_mv/losses/reprojection.py` — reprojection loss (uses original `K, R, t`).
- `configs/train_ray_attention_reproducible.yaml` — reference smoke-training config.

## Proposed code changes

### New files

1. `motionflow_mv/fusion/camera_centric_coordinate_transform.py`
   - Class: `CameraCentricCoordinateTransform`
   - Predicts per-view residual rotation `ΔR` (so(3) parameter), residual translation `Δt`, and a positive per-view depth scale `s`.
   - Inputs: deep per-view features `(N, V, d)` and/or raw 2D+intrinsic descriptor; original `(R, t)`.
   - Outputs: corrected `(R', t')` and optional scale `s`.
   - Initialisation is identity (zero residual, unit scale) so the anchor checkpoint remains warm-startable.

2. `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_camera_centric_model.py`
   - Class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCameraCentric`
   - Subclasses the anchor `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
   - Adds a `CameraCentricCoordinateTransform` instance.
   - Pipeline:
     1. Principal-point / focal correction (existing).
     2. Extract per-frame deep features with existing `_extract_frame_features`.
     3. Run spatio-temporal transformer (existing).
     4. Pool per-view features and predict residual `(ΔR, Δt, s)`.
     5. Apply transform to `(R, t)` to obtain camera-corrected extrinsics.
     6. Triangulate with corrected intrinsics + corrected extrinsics.
     7. Residual MLP refinement (existing).
   - New constructor args:
     - `camera_centric_hidden: int = 64`
     - `max_rot_offset_deg: float = 2.0`
     - `max_trans_offset_m: float = 0.05`
     - `max_scale_delta: float = 0.05`
     - `condition_on_deep_features: bool = True`

3. `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_camera_centric_module.py`
   - Class: `RayAttentionTemporalCrossviewResidualPrincipalPointCameraCentricFusionModule`
   - Thin `FusionModule` wrapper (same pattern as the anchor module) so the new model can be registered in `FUSION_REGISTRY`.

### Modified files

- `motionflow_mv/fusion/__init__.py` — register the new FusionModule (commented by default; user enables after smoke test).
- No changes to existing running experiments or anchor model.

## Training / smoke plan

Use the existing H36M smoke training script with the new model class substituted for the anchor.

- Dataset: H36M smoke split (subjects 1, 5, 6, 7, 8 for train; subject 9 for val) or the synthetic MPI-INF-3DHP smoke set.
- Epochs: 5
- Batch size: 16–32 depending on RTX 4090 memory
- Optimiser: Adam, lr = 1e-3
- Loss: 3D MPJPE + reprojection loss + optional identity regularisation on `(ΔR, Δt, s)`
- Estimated runtime on RTX 4090: ~45–90 minutes for 5 epochs on the smoke dataset.
- Smoke check:
  1. Forward pass on a single batch completes without NaNs.
  2. `ΔR` starts near identity and `s` near 1.
  3. After 1 epoch, 3D MPJPE < baseline-by-epoch (or at least non-degenerate).

## Success metrics

- Primary: clean MPJPE on MPI-INF-3DHP S2/Seq1 ≤ 9.00 mm (target improvement over anchor 9.32 mm).
- Secondary:
  - Relative improvement under simulated calibration perturbation (±5 px principal point, ±2° rotation, ±5 cm translation) ≥ 10 % over anchor.
  - Per-view weight entropy increases for views with larger predicted transform residual (interpretability check).
  - No regression on Human3.6M smoke benchmark.
- Ablations (paper-style):
  - Remove rotation residual only.
  - Remove translation residual only.
  - Remove scale residual only.
  - Condition transform on raw 2D descriptor vs. deep features.

## Risk and fallback

- **Risk:** Learned extrinsics may overfit to the training camera rig or collapse to a degenerate configuration.
  - *Mitigation:* tight bounds on residuals, identity-initialised weights, and an regularisation loss `λ·(‖ΔR‖² + ‖Δt‖² + (s−1)²)`.
- **Risk:** DLT triangulation with corrected extrinsics can become numerically unstable if `s` deviates too far from 1.
  - *Mitigation:* clamp `s ∈ [0.95, 1.05]` via softplus+sigmoid; use the same `weights clamp(min=1e-4)` guard as the anchor.
- **Risk:** Training diverges when combining new transform with principal-point correction.
  - *Mitigation:* freeze principal-point correction for the first 1–2 epochs, then unfreeze.
- **Fallback:** If the camera-centric transform does not outperform the anchor after 5 epochs, freeze it to identity and publish the result as a negative ablation; the branch remains a clean subclass so reverting is one-line.
