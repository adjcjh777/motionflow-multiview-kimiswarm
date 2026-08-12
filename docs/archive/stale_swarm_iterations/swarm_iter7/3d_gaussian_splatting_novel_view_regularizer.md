# Direction 18 — 3D Gaussian Splatting / Novel-View Synthesis Regularizer

## Problem statement

The current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, predicts 3D pose directly from multi-view 2D keypoints but has no rendering-based constraint: a set of 3D joints that projects back consistently across all views is more likely to be geometrically valid. 3D Gaussian splatting offers a differentiable way to turn the predicted 3D skeleton into view-renderable observations, so adding a small auxiliary rendering consistency loss could improve robustness to occlusion, noise, and calibration errors. However, full rasterized Gaussian splatting is expensive, may conflict with the lightweight story of the paper, and is listed as P2 in the plan. The simplest first step is therefore to isolate the regularizer on a CPU-only toy scene and only later hook it into the GPU training loop.

## Simplest concrete next experiment

Add a CPU-only smoke test `experiments/smoke_gaussian_pose_regularizer.py` that:

1. Creates a toy 17-joint skeleton and 4 calibrated cameras.
2. Represents each predicted 3D joint as a small isotropic Gaussian.
3. Projects those Gaussians into each view and renders a low-resolution heatmap.
4. Compares the rendered heatmap against a heatmap built from the observed 2D keypoints.
5. Verifies that a perturbed prediction yields a higher consistency loss than the ground-truth pose.

This validates the formulation before any GPU training is queued.

## Files touched and rough diff

- **New file** `experiments/smoke_gaussian_pose_regularizer.py`
  - `make_circular_cameras()` – simple pinhole rig.
  - `project_points()` – pinhole projection helper.
  - `project_gaussians()` / `render_heatmap()` – tiny 3D→2D Gaussian splatting on CPU.
  - `heatmap_from_keypoints()` – observed keypoint heatmap.
  - `gaussian_pose_regularizer()` – consistency loss between predicted render and observation.
  - `main()` – smoke test with perfect vs. perturbed prediction.

- **Later (GPU, do not run now)** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
  - Add an `--gaussian_reg_weight` argument.
  - After the model produces `pred_3d`, optionally call a new loss `gaussian_splatting_loss(pred_3d, xb[..., :2], K, R, t)`.
  - Add to total loss: `loss = loss + args.gaussian_reg_weight * gauss_loss`.

## Expected success metric

- CPU smoke: `python experiments/smoke_gaussian_pose_regularizer.py` runs in <1 s and reports that the perturbed prediction has a higher loss than the perfect prediction.
- GPU follow-up (queued, not run now): when attached to training, the regularizer should not increase the baseline clean MPJPE by more than 0.2 mm, and it should give a measurable gain on the synthetic occlusion / noisy calibration splits.  A positive result would be clean MPJPE ≤ 9.3 mm and ≥5 % relative improvement on the rot_0.5° / focal_1 % robustness scenarios.

## Resource requirements

- **CPU-only** for the smoke test.
- **GPU** for the follow-up training experiment (queued; the RTX 4090 is currently running the cross-view PP curriculum).

## Command and result

```bash
$ python experiments/smoke_gaussian_pose_regularizer.py
```

Output:

```text
3D Gaussian pose regularizer smoke test
----------------------------------------
[perfect]
  views:             4
  joints:            17
  view_mse:          0.384428
  total_loss:        0.384428
[perturbed]
  views:             4
  joints:            17
  view_mse:          0.493198
  total_loss:        0.493198
----------------------------------------
Smoke test completed successfully.
```

The test confirms the regularizer is numerically stable and sensitive to 3D perturbation.

## Next GPU step (not started)

Once GPU is free, implement `GaussianSplattingLoss` as a PyTorch module in `motionflow_mv/losses/gaussian_splatting.py` and integrate it as an optional auxiliary loss in the principal-point training script.  Because the smoke test is CPU-only, there is no risk to the currently running GPU job.
