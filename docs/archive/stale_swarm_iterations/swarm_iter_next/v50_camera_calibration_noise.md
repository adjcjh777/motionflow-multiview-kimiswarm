# v50 Camera Calibration Noise Robustness (CalNoiseV50)

## One-line idea
Make the v46/v48 triangulation stack robust to noisy/inaccurate camera calibration by injecting synthetic calibration noise during training and adding a small, bounded per-camera correction head that refines intrinsics/extrinsics from reprojection residuals.

## Architecture

A new optional block, `CalNoiseV50`, sits between the v45 triangulation output and the v46 sparse-view head.  During training it (1) perturbs the input camera matrices with synthetic noise, (2) re-triangulates the 3-D pose with the perturbed cameras, and (3) feeds the resulting per-view reprojection residuals into a tiny MLP that predicts bounded updates to focal length, principal point, rotation (axis-angle), and translation.  The correction is applied additively and clamped so the camera parameters cannot drift far from their original values; the corrected cameras are then used for a second triangulation pass.  At inference the module is optional: either disabled (default) or run in iterative refinement mode (K≤2 steps).  The head is identity-initialized where possible and shares its residual-computation logic with the existing v37 reliability code, so it adds only a small number of learnable parameters.

## New config flags

| Flag | Default | Description |
|---|---|---|
| `use_v50_calibration_noise` | `False` | Enable the CalNoiseV50 block. |
| `v50_cal_noise_intrin_std` | `0.05` | Relative std for focal length / principal-point noise. |
| `v50_cal_noise_rot_std_deg` | `0.5` | Rotation noise std in degrees. |
| `v50_cal_noise_trans_std` | `0.01` | Translation noise std in meters. |
| `v50_cal_correction_hidden` | `64` | Hidden size of the per-camera correction MLP. |
| `v50_cal_correction_max_steps` | `2` | Iterative refinement steps during training. |
| `v50_cal_correction_loss_weight` | `0.01` | Weight of the auxiliary residual-reduction loss. |

## Loss term

Primary pose loss remains MPJPE.  We add a gradient-safe auxiliary loss that encourages the correction to reduce reprojection error:

\[
\mathcal{L}_{\text{cal}} = \lambda_{\text{cal}} \cdot \frac{1}{VJ}\sum_{v,j} w_{vj}\,\rho\bigl(\hat{\epsilon}_{vj}^{\text{corrected}} - \hat{\epsilon}_{vj}^{\text{noisy}}\bigr),
\]

where \(\hat{\epsilon}_{vj}\) is the per-view, per-joint reprojection error and \(\rho\) is the Huber loss.  The loss is only active when synthetic noise is injected; with clean calibration it is zero and the correction head is identity.

## Evaluation metric

- `val_MPJPE` and `MPJPE@k` for `k = 2,3,4` from `experiments/eval_variable_views.py`.
- New diagnostic: `cal_mean_reproj_reduction`, the average percentage by which corrected reprojection error drops below the noisy-camera baseline.
- On 3DPW actual-mode, report `MPJPE@2` and `MPJPE@full` with and without the inference-time correction.

## Expected MPJPE impact

On the existing v46-SVG smoke baseline (epoch-1 `val_MPJPE = 32.97 mm`), CalNoiseV50 should have limited full-view impact but should improve sparse-view triangulation where noisy calibration hurts most: expect `MPJPE@2` to drop by **2–4 mm**, with `MPJPE@full` staying within **±1 mm**.  On 3DPW actual-mode, where calibration is noisier, the gain could reach **4–6 mm** on `MPJPE@2`.

## Main risk / mitigations

**Risk: correction head overfits or amplifies calibration drift.** The v21 neural-BA attempt diverged when corrections were unbounded.  Mitigations: clamp each correction term (rotation ≤0.5°, translation ≤10 mm, intrinsics ≤5%), initialize the head to identity/zero, freeze the correction head for the first epoch, and stop gradients from the correction to the base pose features during warmup.  Only enable inference-time refinement after smoke tests show stable `cal_mean_reproj_reduction > 0`.
