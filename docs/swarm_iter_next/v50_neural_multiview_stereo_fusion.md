# v50 Neural Multi-View Stereo Fusion (`NeuralMultiViewStereoFusionV50`)

## One-sentence claim

v50 adds a lightweight, geometry-aware neural stereo-fusion branch to the v25/v45 geometry-fusion backbone. It samples epipolar feature pairs across calibrated views, builds a coarse disparity cost volume, and fuses the resulting multi-view stereo representation with the existing triangulation features, closing the gap between projective triangulation and dense neural stereo matching for sparse-view 3D human pose.

## Architecture description

The module sits after the per-view 2D keypoint/encoder features and before the final pose regression head. For each target view, it projects keypoint rays into a companion view using the known calibration, samples 1-D epipolar feature strips, and stacks them into a small cost volume over a discrete set of depth hypotheses centered on the v25 triangulation depth. A shallow 3-D convnet (or MLP) aggregates the cost volume into a per-joint stereo descriptor, which is concatenated with the v45 adaptive triangulation weights and fed into the existing fusion head. The whole branch is identity-initialised via a residual projection so that, when disabled, behaviour reduces to the v46/v48 baseline.

## New config flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_v50_neural_multiview_stereo_fusion` | `False` | Master on/off switch |
| `v50_nmsf_cost_volume_depths` | `32` | Number of depth hypotheses around the triangulated depth |
| `v50_nmsf_depth_window_m` | `0.5` | Depth search window in meters |
| `v50_nmsf_hidden` | `64` | Hidden dim of the stereo descriptor |
| `v50_nmsf_num_layers` | `2` | Depth of the cost-volume aggregation network |
| `v50_nmsf_dropout` | `0.1` | Dropout in the aggregation MLP |
| `v50_nmsf_views_per_group` | `4` | Max views used to build one stereo cost volume |
| `v50_nmsf_loss_weight` | `0.01` | Weight of the stereo-consistency auxiliary loss |

## Loss term

`L_nmsf = v50_nmsf_loss_weight * MSE(pose_pred_nmsf, pose_gt)`

The auxiliary loss encourages the stereo branch to predict the ground-truth 3D pose directly, giving the branch a clean gradient signal independent of the main pose head. Weight default `0.01` keeps it subordinate to the primary MPJPE loss.

## Evaluation metric

Report `MPJPE@k` for `k = 2, 3, 4, full` via `experiments/eval_variable_views.py`. Additionally log the stereo-branch-only error `MPJPE@k_nmsf` to measure how much the new branch contributes before fusion. Track per-joint uncertainty calibrated against reprojection residual, since the cost-volume sharpness can serve as a stereo confidence proxy.

## Expected MPJPE impact

Local v46-SVG smoke is at `val_MPJPE = 32.97 mm`. Adding the stereo-fusion branch is expected to improve the hardest sparse-view cases where triangulation alone is ill-conditioned. Target: `MPJPE@2` reduced by **2–4 mm**, `MPJPE@3` by **1–2 mm**, and `MPJPE@full` held within **0.5 mm** of the v46/v48 baseline. On the full A800 run this translates to pushing sparse-view generalization closer to the v25 full-view ~17 mm ceiling without regressing full-view accuracy.

## Main risk / mitigations

| Risk | Mitigation |
|------|------------|
| **Cost-volume memory blow-up** | Limit `v50_nmsf_views_per_group` to 4 and down-sample the 1-D epipolar strip to 32 samples; use mixed precision and optional gradient checkpointing |
| **Epipolar sampling is fragile with noisy calibration** | Initialise depth hypotheses around the v25 triangulated point; keep the search window small (`0.5 m`) and clamp sampled coordinates |
| **Stereo branch overfits and ignores triangulation features** | Residual identity initialisation; freeze base model for the first epoch; monitor `MPJPE@k_nmsf` correlation with the fused output |
| **Training instability from extra loss** | Start with `v50_nmsf_loss_weight = 0.0` for 500 steps, then linearly ramp to `0.01` |

## Smoke plan

Use `configs/benchmark_v50_nmsf_smoke.yaml` with `d=64`, `clip_len=9`, `train_samples=500`, warm-start from the best v46-SVG checkpoint. Success gate: `val_MPJPE` finite, `MPJPE@2` lower than v46-SVG smoke by ≥1 mm, no NaN/OOM, and wall-clock time per step increase <15 %.

## Dependencies

v25/v45 geometry fusion, v46 sparse-view generalisation, and the canonical `MPJPE@k` evaluation protocol. Optional but recommended: v37/v39 self-critique reliability, so the cost-volume view groups can be selected by reliability.

## Labels

`experiment`, `P1-next`, `v50`
