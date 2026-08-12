# v50 Lightweight Geometry BA (`LightweightGeometryBAv50`)

## One-sentence idea

Insert a tiny, bounded, differentiable bundle-adjustment step between the v45-AGF triangulation and the v46 sparse-view reliability head, so the model can correct small camera/pose geometry errors from its own reprojection residuals without repeating the v21 neural-BA divergence.

## Architecture

`LightweightGeometryBAv50` refines the 3-D pose hypothesis produced by v45-AGF using one or two closed-form re-triangulation iterations. It takes the current 3-D pose `P`, per-view 2-D keypoints `x` and their v46/v37 reliability weights, and the camera parameters `(K, R, t)`.

1. **Residual encoding.** For each view, compute normalized reprojection residuals `r = (x - π(P; K,R,t)) * w`, where `w` is the v46 reliability weight. A per-view two-layer MLP maps the flattened residual vector to a camera update `(Δφ, Δt)` and a joint-level pose offset `δP`.
2. **Bounded update.** Rotation is updated via `R' = R * exp([Δφ]×)` with `|Δφ|` clipped to a small maximum angle. Translation and pose offsets are clamped to centimeter-scale ranges. Identity initialization means `Δφ=0`, `Δt=0`, `δP=0`, so the block is transparent at the start of training.
3. **Re-triangulation.** The corrected cameras and pose are used in a single DLT/linear triangulation step to produce the refined 3-D output `P'` that is fed forward.

The block is intentionally tiny (~4 k parameters), runs in closed form, and does not use an unrolled solver that could amplify gradients.

## Config flags and defaults

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `use_v50_lightweight_geometry_ba` | bool | `False` | Enable the module |
| `v50_lgba_hidden` | int | `64` | Hidden dim of the per-view residual MLP |
| `v50_lgba_num_iter` | int | `2` | Max re-triangulation iterations (cap at 2) |
| `v50_lgba_max_rot_deg` | float | `1.0` | Hard bound on per-update camera rotation |
| `v50_lgba_max_trans_m` | float | `0.05` | Hard bound on per-update camera translation (meters) |
| `v50_lgba_max_pose_m` | float | `0.05` | Hard bound on per-joint pose correction |
| `v50_lgba_update_cameras` | bool | `True` | Allow camera correction |
| `v50_lgba_update_pose` | bool | `True` | Allow joint pose correction |

## Loss term(s)

- **`v50_lgba_reproj_weight`: float, default `0.01`**
  Adds the weighted reprojection error of the refined 3-D pose:
  `L = λ * Σ_v w_v || π_v(P') - x_v ||²`.
  This gives the geometry block a direct signal independent of the final MPJPE loss.

- **`v50_lgba_reg_weight`: float, default `1e-4`**
  Regularizes the magnitude of the updates:
  `L_reg = λ_reg (||Δφ||² + ||Δt||² + ||δP||²)`.
  Keeps the corrections small and prevents the v21-style unbounded drift.

## Evaluation metric

Report alongside existing `val_MPJPE@full` and `MPJPE@k` from `experiments/eval_variable_views.py`:

- `mean_reprojection_error_px` (after refinement)
- `mean_camera_update_magnitude` (translation + rotation magnitude averaged over views)
- `mean_pose_correction_magnitude` (per-joint correction before triangulation)

## Expected MPJPE impact

- `val_MPJPE@full`: **-0.5 to -1.5 mm**
- `MPJPE@2`: **-2 to -4 mm** (sparse views gain most from better geometry)
- `MPJPE@3/4`: **-1 to -2 mm**

These numbers assume the block is warm-started from a stable v46/v47 checkpoint on the local RTX 4090 smoke config (`d=64`, `clip_len=9`, `train_samples=500`).

## Main risk / mitigations

**Risk: re-introducing the v21 neural-BA divergence.** Even bounded camera updates can drift if the training signal is noisy or the refinement iterations are unrolled too deeply.

**Mitigations:**
1. Hard clip all updates (rotation ≤ 1°, translation/pose ≤ 5 cm).
2. Identity initialization so the block is transparent at start.
3. Cap iterations at 2 and use a closed-form DLT step, not an iterative optimizer.
4. Add `v50_lgba_reg_weight` to penalize large updates.
5. Freeze the block for the first epoch when warm-starting from v46/v47, then unfreeze.
6. Gradient clipping at the module output.

## Smoke plan

Add a smoke config `configs/benchmark_v50_lgba_smoke.yaml` that enables the module on top of the v46-SVG smoke baseline. Smoke goal: `val_MPJPE@full` within 1 mm of v46-SVG and `MPJPE@2` improved by at least 2 mm; no NaN/OOM; reprojection error decreases.
