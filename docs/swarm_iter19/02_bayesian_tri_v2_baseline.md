# 02 Bayesian Tri v2 Baseline

## Summary

This subtask reviews the **Bayesian Tri v2** baseline: a multi-view 3D pose
estimator that adds anisotropic image-space covariance, adaptive Gauss-Newton
refinement, and an epipolar consistency loss to the principal-point cross-view
residual anchor. It matters because the project's current best MPI-INF-3DHP
results (single model and ensemble) are built on this exact variant, so its
health, reproducibility, and remaining failure modes bound the next iteration.

## Current state

- **Model code**: `RayAttentionFusionModelBayesianTriV2` is defined in
  `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py:371-499`.
  It inherits the PP-corrected cross-view temporal transformer and swaps the
  per-joint DLT loop for the fully batched `triangulate_dlt_batched_lstsq` in
  `motionflow_mv/fusion/triangulation.py:106-155`.
- **Training entry points**:
  - `experiments/train_bayesian_tri_v2_smoke.py:154-257` runs a tiny synthetic
    CPU smoke test.
  - `experiments/train_bayesian_tri_v2_full_mpiinf3dhp.py:14-48` launches the
    full MPI-INF-3DHP protocol (`model_type=bayesian_tri_v2`).
- **Production recipes**:
  - `scripts/run_bayesian_tri_v2_stabilized_wsl.sh` uses cosine LR, 3-epoch
    warmup, gradient clipping, and the extended intrinsics curriculum.
  - `scripts/run_bayesian_tri_v2_aug_wsl.sh` adds per-view noise, joint dropout,
    and view dropout on top of the stabilized recipe.
- **Best results on MPI-INF-3DHP S2/Seq1**:
  - Single model: `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth` at
    **9.03 mm** MPJPE (`outputs/bayesian_tri_v2_stabilized_mpiinf3dhp_eval.json`).
  - Two-model ensemble: **8.35 mm** MPJPE
    (`outputs/bayesian_tri_v2_ensemble_2_eval.json`).
- **Tests pass**: `tests/test_bayesian_tri_v2_batched_dlt.py` and
  `tests/test_bayesian_tri_v2_batched_dlt_v2.py` report 16 passed, 1 skipped,
  and the smoke trainer completes on CPU.

## Key findings

1. **The stabilized training recipe is essential.** The original large-scale run
   (`scripts/run_bayesian_tri_v2_large_scale_wsl.sh`) had no cosine LR, warmup,
   or gradient clipping; its validation MPJPE collapsed from 9.71 mm at epoch 8
   to ~40 mm from epoch 13 onward (`outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.log:9-38`).
   The stabilized recipe avoids this and reaches the current single-model best.

2. **The batched DLT is numerically sound.** Unit tests cover two-view minimum
   cases, weight invariance, gradient flow, and float32/float64 consistency
   (`tests/test_bayesian_tri_v2_batched_dlt_v2.py:66-196`).

3. **Model outputs are correctly wired.** The forward pass returns
   `(pred_3d, weights, pp_delta, epi_loss)`, and the trainer consumes the
   auxiliary epipolar loss at
   `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py:649-651`.

4. **The ensemble evaluation path is fragile.** The default `d=64` in
  `experiments/prototypes/eval_ensemble_checkpoints.py:52` causes size-mismatch
   failures when loading the d=128 anchor checkpoints, as seen in
   `outputs/bayesian_tri_v2_ensemble_eval.log:12-84`.

5. **Error is concentrated in the limbs.** Per-joint MPJPE in the 8.35 mm
   ensemble is ~6.9–8.9 mm on trunk joints but rises above 10 mm on the lower
   limbs (`outputs/bayesian_tri_v2_ensemble_2_eval.json:130-158`).

## Recommendations

1. **Make the stabilized recipe the default.** Deprecate or remove the old
   `run_bayesian_tri_v2_large_scale_wsl.sh` and document the stabilized flags
   (`--lr 3e-4 --lr_cosine --lr_warmup_epochs 3 --grad_clip_norm 1.0`) as the
   Bayesian Tri v2 baseline.

2. **Fix and lock the ensemble evaluation command.** Update the ensemble eval
   script to pass `--d 128 --residual_hidden 256 --n_st_layers 3` by default,
   or derive them from a recipe file, so the 8.35 mm result is reproducible.

3. **Run repeated seeds.** Re-train `run_bayesian_tri_v2_stabilized_wsl.sh`
   with seeds 0–2 to quantify single-model variance before paper submission.

4. **Target limb error next.** Add a skeleton-aware residual refiner or
   kinematic-chain post-processing to the Bayesian Tri v2 head; the per-joint
   error map shows trunk accuracy is already saturated.

## Open questions

- Does the 8.35 mm ensemble hold on the official MPI-INF-3DHP **test subjects
  TS1–TS6**? All current numbers are on S2/Seq1 validation only.
- Is the no-graph ablation of OmniMultiViewFusionV2
  (`outputs/omniview_fusion_v2_d128_no_graph.log`) improving the baseline or is
  it orthogonal work?
- Can the uncertainty-aware `ensemble_inference_v2.py` aggregation
  (`docs/swarm_iter18/P09_ensemble_inference_v2.md`) beat the current 8.35 mm
  uniform average?
- Does the current model's robustness under -D view dropout and joint occlusion
  (18.15 mm / 16.99 mm at 30%) improve if limb-focused refinement is added?
