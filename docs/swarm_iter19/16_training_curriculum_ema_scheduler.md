# 16 — Training Curriculum, EMA, and LR Scheduler

## Summary

This subtask audits the training infrastructure that supports the current OmniMultiViewFusionV2 run: the reusable `TrainerV2` loop, its Exponential Moving Average (EMA) and warmup+cosine LR scheduler, and the camera-perturbation curriculum used by earlier trainers. Getting these pieces right is high-leverage because the no-graph ablation is already running and any scheduler/EMA/curriculum bug will affect the final MPI-INF-3DHP numbers.

## Current state

* `motionflow_mv/training/trainer_v2.py` already provides a clean, tested `TrainerV2` (lines 236-471) with:
  * `_WarmupCosineLR` (lines 35-85) - linear warmup then cosine decay.
  * `EMA` class (lines 150-229) - shadow weights, `apply_shadow`/`restore`, and checkpoint round-trip.
  * Gradient clipping, CPU-safe AMP, and `MultiViewPoseTrainerV2` convenience wrapper.
* `experiments/train_omniview_fusion_v2_mpiinf3dhp.py` wires the above together (lines 674-692) and adds a staged warm-start freeze/unfreeze (lines 697-718).
* `motionflow_mv/calibration/camera_perturbation_curriculum.py` contains mature curriculum schedules (`extended_curriculum`, `extended_intrinsics_curriculum`, cosine anneal) with unit tests in `tests/test_camera_perturbation_curriculum.py`.
* The running no-graph ablation (`scripts/run_omniview_fusion_v2_full_wsl.sh`) is using `--lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --ema_decay 0.999`.

## Key findings

1. **Omni trainer does not use the camera-perturbation curriculum.** `train_omniview_fusion_v2_mpiinf3dhp.py` has no `cam_aug_*` arguments and never calls `perturb_cameras_with_delta`. The older `train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` (lines 278-589) already integrates it, so the gap is simply that the new trainer has not been wired yet. This is the single biggest missing piece for robustness.
2. **EMA itself does not need bias correction, but its checkpoint identity was ambiguous.** The shadow is initialized from the current model rather than zero, so its recursive weights already sum to one. Dividing by `1 - decay^step` would be incorrect. The actual gap was that validation used EMA while external evaluation loaded raw `model` parameters; iteration 20 separates raw `model` from `eval_model`.
3. **Scheduler is epoch-level, not step-level.** `_WarmupCosineLR.get_lr` uses `self.last_epoch`, so LR changes only at epoch boundaries. For short epochs and large batches this is coarse; the current script calls `step_scheduler()` once per epoch (trainer_v2.py:439), which is consistent but may not be optimal for 30-epoch runs.
4. **Warm-start freeze loop is outside `fit` and duplicates scheduler stepping.** In `train_omniview_fusion_v2_mpiinf3dhp.py` lines 705-716 the freeze phase manually calls `trainer.epoch += 1`, `train_epoch`, `evaluate`, and `step_scheduler`. The manual epoch increment was only just fixed in commit `7eb0674`, so this path is still lightly tested.
5. **Validation during freeze uses EMA.** The freeze loop calls `trainer.evaluate(...)`, and that method applies/restores the EMA shadow internally when `ema_eval=True`. The earlier claim that these values came from live weights was incorrect.

## Recommendations

1. **Wire the camera-perturbation curriculum into the Omni trainer.** Add `cam_aug_*` CLI flags to `train_omniview_fusion_v2_mpiinf3dhp.py` and call `perturb_cameras_with_delta` on `K, R, t` before the forward pass, mirroring `train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py:584-589`. Start with `extended_curriculum`, `--cam_aug_warmup_epochs 2`, and the same ranges as the proven prototype (`run_bayesian_tri_v2_stabilized_wsl.sh`).
2. **Keep EMA validation and checkpoint identity explicit.** Save raw training weights separately from the exact evaluation weights, and make evaluation/warm-start loaders use the latter. Do not add zero-init bias correction to this non-zero-initialized EMA.
3. **Expose `ema_update_every` in the Omni script.** The trainer already supports it (trainer_v2.py:277); the script just needs to pass it through. This lets us test whether less frequent EMA updates change final MPJPE.
4. **Keep the epoch-level scheduler for now, but log LR.** Add `lr` to the per-epoch metrics so the decay can be verified in the log. Do not move to step-level scheduling until the no-graph ablation finishes and we have empirical evidence it is needed.
5. **Add a CPU test for the staged freeze path.** Extend `tests/test_train_omniview_fusion_v2_smoke.py` to exercise `warm_start_freeze_epochs > 0` with a synthetic checkpoint so the manual epoch-increment behavior is guarded by CI.

## Open questions

* Does the camera-perturbation curriculum improve OmniMultiViewFusionV2 clean accuracy, or only robustness? We need a d=48 smoke with and without it.
* Does `update_every > 1` improve the final checkpoint over the current `decay=0.999` every-step update?
* Is the epoch-level LR schedule limiting convergence for the 30-epoch Omni run, or is 3-epoch warmup + cosine sufficient?
* How does the staged freeze phase affect EMA initialization? The shadow is initialized from the warm-started weights and then frozen parameters are updated only for new heads for 5 epochs; we should verify this does not bias the shadow mean.
