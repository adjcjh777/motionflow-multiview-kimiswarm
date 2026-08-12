# Subtask 17: Camera Perturbation + Synthetic Occlusion

## Summary

This subtask covers the two main robustness axes that are cheap to simulate but still under-used in the current training loop: **camera calibration perturbations** (intrinsics + extrinsics) and **synthetic joint/view occlusion**. Both are prerequisites for pushing the current 8.35 mm Bayesian Tri v2 ensemble toward a model that is robust to real-world sensor errors and partial visibility. The code for both augmentations is in place and CPU-tested; the gap is integration into the active trainers and the robustness-evaluation harness.

## Current state

- **Camera-perturbation curriculum** is implemented in `motionflow_mv/calibration/camera_perturbation_curriculum.py` (`extended_camera_perturbation_schedule`, line 9; `extended_camera_perturbation_schedule_with_anneal`, line 81; `schedule_from_args`, line 122). It supports flat, extrinsic-curriculum, intrinsics-curriculum, and combined extended schedules.
- **Low-level camera perturbations** are in `motionflow_mv/calibration/perturb.py` (`perturb_cameras_with_delta`, line 187), handling per-view rotation, translation, focal-length, and principal-point noise.
- **Synthetic group occlusion** is implemented in `motionflow_mv/data/synthetic_occlusion_aug.py` (`SyntheticJointOcclusionAugmenter`, line 212) and supports anatomically coherent joint groups for both H36M-17 and MPI-INF-3DHP-28 skeletons, with optional temporal consistency and reproducible `state_dict`/`load_state_dict`.
- A prototype version also exists in `motionflow_mv/data/prototypes/synthetic_joint_occlusion.py`.
- Smoke tests pass:
  - `tests/test_camera_perturbation_curriculum.py` (6 passed)
  - `tests/test_synthetic_occlusion_aug.py`
  - `tests/test_synthetic_joint_occlusion.py`
  - `experiments/eval_occlusion_robustness.py` CPU smoke check

What does **not** yet work end-to-end:
- The active `OmniMultiViewFusionV2` trainer (`experiments/train_omniview_fusion_v2_mpiinf3dhp.py`) only applies `augment_clip` (2-D pixel noise, confidence dropout, view dropout; line 239) and does **not** use the extended camera-perturbation curriculum or the group-level synthetic occluder.
- The robustness-evaluation scripts (`experiments/run_robustness_matrix.py`, `experiments/eval_occlusion_robustness.py`) test only plain random view/joint dropout and a single principal-point perturbation, not the extended calibration curriculum.

## Key findings

1. **Camera curriculum exists but is not wired into the main trainer.** `train_omniview_fusion_v2_mpiinf3dhp.py` never calls `perturb_cameras_with_delta` or `extended_camera_perturbation_schedule`; it only perturbs 2-D keypoints and confidences inside `augment_clip`.
2. **Group-level synthetic occlusion is isolated.** `SyntheticJointOcclusionAugmenter` is fully tested, but no active dataloader or trainer imports it. The visibility head in `OmniMultiViewFusionV2` is trained only against the real observation mask (`x[..., 2] > 0`, line 412), missing the opportunity to use synthetic occlusion masks as auxiliary visibility labels.
3. **Legacy trainers use partial perturbation only.** `experiments/train_crossview_residual_uncertainty_mpiinf3dhp.py` (line 260) ramps extrinsic noise via a hard-coded `extrinsic_curriculum` branch but leaves intrinsics (`focal`, `pp`) flat. The extended curriculum with cosine annealing is unused.
4. **Evaluation coverage is narrow.** `experiments/eval_perturb_model_mpiinf3dhp.py` (lines 207-224) evaluates many conditions, but the matrix is not run in CI. `run_robustness_matrix.py` (lines 62-72) only tests `cam_aug_pp` and view dropout, not rotation/translation/focal perturbations.

## Recommendations

1. **Wire the extended camera curriculum into `train_omniview_fusion_v2_mpiinf3dhp.py`.**
   - Add CLI args matching `schedule_from_args` (`cam_aug_schedule`, `cam_aug_rot`, `cam_aug_trans`, `cam_aug_focal`, `cam_aug_pp`, ramp epochs, warmup).
   - In the training loop, call `schedule_from_args(epoch, args, total_epochs=args.epochs)` and apply `perturb_cameras_with_delta(K, R, t, ...)` before the forward pass.
   - Start with a smoke run on the existing `--smoke` synthetic dataset to verify no shape/broadcast issues.

2. **Integrate `SyntheticJointOcclusionAugmenter` as a training augmentation.**
   - Instantiate it in `train_omniview_fusion_v2_mpiinf3dhp.py` (use `skeleton="mpiinf3dhp_28"` for MPI data, `temporal_consistency=True`, small `group_rate` ~0.1-0.2).
   - Apply it inside `augment_clip` or directly after loading the clip, then feed the same occluded `x` to the model and use the synthetic occlusion mask as an auxiliary target for the visibility BCE loss.

3. **Run a small ablation before committing GPU time.**
   - Use the existing `experiments/eval_perturb_model_mpiinf3dhp.py` or extend `run_robustness_matrix.py` with `extended_curriculum` conditions (rot 0.5/1.0 deg, trans 5/10 mm, focal 1/2 %, pp 3/5 px).
   - Smoke-test the combined clean+perturbed evaluation path on a tiny .npz to ensure JSON/Markdown outputs still parse.

4. **Avoid duplication.**
   - Decide whether to keep `motionflow_mv/data/prototypes/synthetic_joint_occlusion.py` or delete it; `motionflow_mv/data/synthetic_occlusion_aug.py` is the newer, better-documented version and already has the canonical tests.

## Open questions

- What is the right perturbation magnitude for the active 8.35 mm pipeline? Extended curriculum defaults (rot 2.0 deg, trans 0.02 m, focal 5 %, pp 10 px) may be too aggressive and should be calibrated against real camera-error statistics from WebBridge/H36M.
- Does group-level occlusion actually improve MPI-INF-3DHP test-set MPJPE, or only robustness metrics? Need an ablation comparing `group_rate` to the existing per-joint `confidence_dropout`.
- Should synthetic occlusion masks be used as hard visibility targets for the visibility head, or only to up-weight uncertain joints? Empirical comparison is needed.
