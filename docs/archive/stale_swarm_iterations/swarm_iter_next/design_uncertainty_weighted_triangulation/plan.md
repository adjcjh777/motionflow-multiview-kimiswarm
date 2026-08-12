# Design Plan: Uncertainty-Weighted Triangulation (v2)

## Goal
Implement an uncertainty-weighted variant of the temporal residual model that
predicts a per-view log-variance for each joint, uses the predicted precision as
the DLT triangulation weight, and supervises the uncertainties with a
reprojection negative log-likelihood (NLL) loss.

## Model: `RayAttentionFusionModelTemporalUncertaintyV2`
- **Location**: `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py`
- **Base**: `RayAttentionFusionModelTemporalResidualV2`
  - Inherits the V4-normalised camera embedding (scale/resolution invariant).
  - Inherits the residual refinement head and optional reprojection gate.
- **New components**:
  1. `uncertainty_head`: `nn.Linear(d, 1)` predicting `log_var` per (view, joint).
  2. Clamping of `log_var` to `[-10, 10]` for numerical stability.
  3. Weight computation: `w = confidence * exp(-log_var)`.
  4. Reprojection NLL auxiliary loss: `0.5 * (err^2 / var + log_var)`.

## Training Script: `experiments/train_uncertainty_v2_mpiinf3dhp.py`
- Follows the same data-loading/collate pattern as the residual train script.
- Model returns `(pred_3d, weights, log_var, nll_loss)`.
- Loss: `MSE(pred_3d, gt) + reproj + bone + nll_loss`.
- Supports all existing flags (`reproj_weight`, `bone_weight`, `use_reproj_gate`,
  camera augmentation, etc.).

## Validation / Smoke Test
1. Model-level sanity check: forward/backward on synthetic `(B, T, V, J, 3)`
   inputs and a 4-view circular rig.
2. Training-script smoke run on a small H36M subset for one epoch.

## Expected Impact
- Uncertainty weights give the model a learned mechanism to down-weight noisy
  or occluded views during triangulation.
- The reprojection NLL should make predicted uncertainties correlate with
  actual reprojection error, improving robustness.
- The v2 camera embedding preserves compatibility with V4 checkpoints and
  cross-scale generalisation.

## Risks / Open Questions
- The NLL loss can dominate early training if `uncertainty_weight` is too high;
  default is set to `0.1` to keep it subordinate to the 3D MSE loss.
- Clamping `log_var` is required for stability but may cap the dynamic range
  of the predicted weights.
- Full MPI-INF-3DHP validation and comparison against the 10.46 mm baseline is
  left for the broader swarm training run.
