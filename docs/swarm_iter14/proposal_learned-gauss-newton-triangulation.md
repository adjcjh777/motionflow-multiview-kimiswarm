# Learned Gauss-Newton Triangulation for the PP-Corrected Anchor

**Date:** 2026-08-06  
**Author:** MotionFlow-MultiView iter14 swarm  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  

---

## 1. Problem

The current anchor triangulates with a single weighted-DLT step, which is a closed-form least-squares solution and therefore cannot iteratively refine the 3-D estimate using the learned per-view weights or reprojection residuals.

## 2. Hypothesis

Adding a differentiable Gauss-Newton (GN) refinement after DLT—driven by the same learned per-view weights and fully back-propagable—will reduce reprojection error and lower MPJPE/PA-MPJPE on clean and corrupted cameras without changing the attention backbone or the PP-correction head.

## 3. Method

### 3.1 Architecture change

Create a new model that subclasses the anchor and replaces the DLT-only triangulation with **DLT initialization + iterative GN refinement**:

- **New model file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_learned_tri_model.py`
  - Class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointLearnedTri`
  - Inherits from `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` (the 9.32 mm anchor).
  - Re-uses the existing `_triangulate_weighted_gauss_newton` helper from `motionflow_mv/fusion/ray_attention_temporal_learned_tri_v1.py` (or duplicates it in the new file to avoid cross-model coupling).
  - Override the forward/triangulate logic so that after `PrincipalPointCorrection` and weight prediction:
    1. Compute DLT estimate `X_dlt` with corrected intrinsics.
    2. Refine `X_dlt` with `num_iters` GN steps on the weighted reprojection error.
    3. Feed the GN-refined estimate into the existing residual MLP.
  - Expose two new hyperparameters: `gn_iters` (default 3) and `gn_damping` (default `1e-6`).

- **New training script:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_learned_tri_mpiinf3dhp.py`
  - Copy of `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`, but instantiates `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointLearnedTri`.
  - Add CLI flags: `--gn_iters` (default 3), `--gn_damping` (default `1e-6`).
  - Keep all existing PP/focal augmentation and the direct intrinsics loss path so the experiment isolates the triangulation change.

### 3.2 Loss / data changes

- No new dataset is required; reuse the same MPI-INF-3DHP `.npz` clips as the anchor.
- Loss remains `mpjpe_loss + pp_loss_weight * intrinsics_loss` (or whatever the anchor trainer already uses).
- The GN step itself introduces no extra loss term; gradients flow from the final 3-D MPJPE back through the GN iterations, the predicted weights, and the PP-correction head.

### 3.3 Code modules touched

| File | Action |
|------|--------|
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_learned_tri_model.py` | Create: new model subclass with DLT+GN triangulation. |
| `experiments/train_ray_attention_temporal_crossview_residual_principal_point_learned_tri_mpiinf3dhp.py` | Create: trainer mirroring the anchor trainer. |
| `motionflow_mv/fusion/__init__.py` | Add the new class to exports. |

## 4. Smoke-Test Plan

Run a short 5-epoch smoke on MPI-INF-3DHP subject 1, sequence 1, with a tiny model to verify finite loss, stable gradients, and no NaNs.

```bash
conda run -n mf python experiments/train_ray_attention_temporal_crossview_residual_principal_point_learned_tri_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
    --gn_iters 3 --gn_damping 1e-6 \
    --epochs 5 --batch_size 4 --lr 1e-3 \
    --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01
```

**Pass/fail criteria:**

- **Pass:** Training completes 5 epochs without NaN/Inf; final val MPJPE ≤ 15 mm (smoke-sized model, so absolute accuracy is not the target).
- **Pass:** Gradient norms are finite for all model parameters, including the GN block.
- **Pass:** Runtime per epoch is within 2× of the anchor PP trainer on the same smoke config.
- **Fail:** Any crash, NaN loss, or epoch time > 3× anchor.

## 5. Evaluation Plan

After the smoke, evaluate the trained checkpoint with the existing full-metrics and robustness scripts:

- **Accuracy metrics:** MPJPE, PA-MPJPE, PCK@50/100/150, AUC on MPI-INF-3DHP S2/Seq1.
  - Script: `experiments/eval_full_metrics.py --model learned_tri_pp --checkpoint <path>`.
  - Target: clean val MPJPE ≤ 9.6 mm (within ~3% of the 9.32 mm anchor).
- **SOTA comparison:** Run `experiments/compare_sota_baselines.py` to confirm the new model still beats DLT/IRLS by a wide margin.
- **Robustness matrix:** Run `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` with the new checkpoint.
  - Target: clean MPJPE ≤ 9.6 mm and no moderate-severity corruption degrades > 30% relative to clean.
- **Ablation:** Compare `gn_iters=0` (equivalent to the anchor) vs. `gn_iters=3` on the same seed to attribute any gain specifically to the GN step.

## 6. Estimated GPU/CPU Cost on RTX 4090

- **Smoke (5 epochs, d=32, residual_hidden=64, ~500 clips):** ~15–25 minutes on a single RTX 4090; CPU-only data loading, no multi-GPU.
- **Full validation run (S2/Seq1):** ~2–5 minutes on RTX 4090 inference.
- **Full training if smoke passes (30 epochs, d=64, residual_hidden=128):** ~6–10 hours on RTX 4090, or roughly 1.2–1.5× the anchor trainer due to the GN solve.

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|------|----------------------|
| GN iterations are unstable or produce NaNs when cameras are perturbed. | Increase `gn_damping` to `1e-4` or clamp the GN update magnitude; if still unstable, set `gn_iters=0` to fall back to the anchor DLT behavior. |
| GN adds too much latency (e.g., >50% slower per epoch). | Reduce `gn_iters` to 1, or make GN optional via a flag; keep the model as a fast-path variant. |
| No accuracy improvement over DLT because the residual MLP already absorbs reprojection residuals. | Run the ablation `gn_iters=0` vs. `gn_iters=3`; if no gain, document that the residual head is sufficient and archive the experiment without merging. |
| The existing `_triangulate_weighted_gauss_newton` helper has shape assumptions that conflict with the PP-corrected K. | Re-implement the GN step inside the new model file with the exact `(B*T, V, J, ...)` shapes used by the anchor. |
