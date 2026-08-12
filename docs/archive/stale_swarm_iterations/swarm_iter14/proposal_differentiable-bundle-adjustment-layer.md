# Differentiable Bundle Adjustment Layer for End-to-End Camera + Pose Refinement

## 1. Problem

The current `PrincipalPointCorrection` head only sees camera-noise gradients through the downstream 3D pose loss, so it saturates and cannot correct extrinsic errors or joint-level pose drift, leaving the 9.32 mm anchor fragile to the `cxcy` and rotation perturbations seen in the iter13 robustness matrix.

## 2. Hypothesis

Adding a lightweight, fully differentiable bundle-adjustment (DBA) layer after DLT triangulation—jointly refining the 3D pose and the camera parameters via reprojection residuals—will give the network explicit geometric feedback, improving both clean accuracy and calibration-robustness without a larger model.

## 3. Method

### 3.1 New module: `motionflow_mv/fusion/differentiable_bundle_adjustment.py`

Create a `DifferentiableBundleAdjustment` layer that performs a fixed small number of Gauss-Newton / Levenberg-Marquardt iterations entirely in PyTorch:

- **Parameters per sample (after DLT):**
  - `X` (B·T, J, 3): per-joint 3D pose, initialized to the residual-refined pose.
  - `ξ` (B·T, V, 6): per-view SE(3) camera perturbation, initialized to zero.
  - `κ` (B·T, V, 4): per-view intrinsic perturbation (Δcx, Δcy, Δfx/fx, Δfy/fy), initialized to zero (optional, gated by `correct_intrinsics`).
- **Objective per iteration:**
  ```
  L_reproj = Σ_v,w  w_v,j · ρ(π(P_v(ξ,κ), X_j) − x_v,j)
  L_reg    = λ_pos · ||X − X_init||² + λ_cam · ||ξ||² + λ_int · ||κ||²
  ```
  where `ρ` is a Huber robust loss, `P_v` is the projection matrix for view `v`, and weights come from the attention-based weight head.
- **Update:** Compute the reprojection-error Jacobian via `torch.autograd.grad`, form the damped normal equations `JᵀJ + μI`, and solve with the SVD pseudoinverse; the step is truncated with `tanh` bounds so the layer stays near identity at init.
- **Output:** refined pose `X*` and refined projection matrices `P*`.

Implementation must be batched, avoid in-place ops that break the autograd graph, and run on GPU.

### 3.2 New model: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_ba_model.py`

Subclass `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`:

- Keep the existing PP correction head and residual MLP unchanged.
- After the residual MLP output, pass `pred_3d`, `points_2d`, `K_corrected`, `R`, `t`, and `weights` into the new `DifferentiableBundleAdjustment` layer.
- Return the BA-refined pose plus the pre-BA pose as an auxiliary output for ablation.
- Add constructor flags: `ba_iterations` (default 2), `ba_damping` (default 1e-2), `ba_correct_cameras` (default True), `ba_correct_intrinsics` (default False for smoke, True for full).

### 3.3 New training script: `experiments/train_differentiable_ba_smoke_mpiinf3dhp.py`

- Copy the structure of `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`.
- Instantiate the new BA model.
- Losses:
  - `L_pose = MPJPE(pred_3d_ba, gt_3d)`
  - `L_reproj = Huber(π_ba, x_2d)` with observation confidence weighting
  - `L_reg = λ_cam · ||ξ||² + λ_int · ||κ||² + λ_pos · ||X_ba − X_init||²`
  - `L_total = L_pose + α L_reproj + L_reg`
- Use the existing intrinsics curriculum from `motionflow_mv/calibration/perturb.py` (start with small PP noise, ramp over epochs).

### 3.4 Evaluation hook (read-only, no existing code changes)

No existing files are modified for the smoke. If the smoke passes, the only downstream integration is to append the new model string to `MODEL_CLASSES` in `experiments/eval_full_metrics.py`:

```python
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_ba_model import (
    RayAttentionFusionModelTemporalCrossviewResidualBA,
)

MODEL_CLASSES = {
    ...,
    "crossview_residual_pp_ba": RayAttentionFusionModelTemporalCrossviewResidualBA,
}
```

## 4. Smoke-Test Plan

Run a 3–5 epoch smoke on MPI-INF-3DHP S1/Seq1 (or the same 500-sample subset used by the factorized ST+PP smoke):

- **Config:** `d=64, n_views=4, clip_len=9, batch_size=4, ba_iterations=2, ba_damping=1e-2`, pose + cameras refined, intrinsics frozen.
- **Pass criteria:**
  1. Training completes with no NaN/Inf and finite val MPJPE.
  2. Average Huber reprojection error on the val set is at least 10 % lower for the BA output than for the pre-BA residual output.
  3. Clean val MPJPE ≤ 9.8 mm (≤ 0.5 mm above the 9.32 mm anchor).
  4. Pearson correlation between predicted camera perturbation `ξ` and the injected extrinsic perturbation is ≥ 0.2.
- **Fail criteria:** any NaN/Inf, val MPJPE > 30 mm, reprojection-error reduction < 5 %, or correlation < 0.1.

## 5. Evaluation Plan

1. **Clean metrics:**
   - Run `experiments/eval_full_metrics.py --model crossview_residual_pp_ba --checkpoint <path> --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`.
   - Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC in mm.
   - Compare against the iter13 anchor (9.32 / 5.37 mm).

2. **Calibration-robustness metrics:**
   - Extend `experiments/eval_perturb_model_mpiinf3dhp.py` to load the BA model and report metrics under:
     - focal ±2 %
     - principal-point ±5 px
     - rotation ±0.5°
     - translation ±10 mm
   - Track reprojection error before/after BA and the mean magnitude of `ξ` / `κ`.

3. **Ablation:** compare `pp_ba` vs. `pp_only` (identical network, `ba_iterations=0`) to isolate the effect of the DBA layer.

## 6. Estimated GPU/CPU Cost on RTX 4090

- **Smoke (3–5 epochs, 500 samples):** ~20–35 minutes on a single RTX 4090. The DBA layer adds ~20–30 % compute per iteration because of the extra Jacobian/back-substitution, but the smoke uses only 2 iterations.
- **Memory:** ~1–2 GB extra VRAM for the per-iteration Jacobian buffers; total training should still fit comfortably in 24 GB.
- **Full eval:** <10 minutes on CPU/GPU (mostly inference and metric computation).

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|------|-----------------------|
| NaNs from ill-conditioned `JᵀJ` in the DBA solver | Add learned/fixed damping `μI`, use SVD pseudoinverse, clamp updates, and initialize `ξ, κ` near zero. |
| Camera refinements collapse to a trivial solution or overfit | Strong L2 regularization toward the initial cameras; freeze camera refinement for the first epoch (curriculum) and only refine pose. |
| DBA iterations slow training too much | Smoke uses `ba_iterations=2`; fallback to pose-only BA or reduce to 1 iteration. |
| BA output does not beat PP baseline after smoke | Abandon camera refinement, keep the direct intrinsics-loss + curriculum fix from iter13 (#2 in `swarm_iter13_next_iteration_synthesis.md`), and use the BA layer only to refine the 3D pose. |
| Difficulty debugging the autograd graph in the solver | Provide a standalone unit test in `tests/test_differentiable_bundle_adjustment.py` with toy 2-view/3-joint data and known ground-truth cameras. |
