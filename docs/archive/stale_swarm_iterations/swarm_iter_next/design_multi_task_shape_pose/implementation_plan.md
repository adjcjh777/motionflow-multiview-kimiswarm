# Implementation Plan — Multi-Task Shape & Pose

## Phase 1: Add the SMPL shape/pose head (1 day)

1. Create `motionflow_mv/fusion/smpl_shape_pose_head.py` with `SMPLShapePoseHead`.
   - Inputs: pooled feature `(B*T, d)`; optional triangulated pelvis `(B*T, 3)`.
   - Outputs: `betas (1, 10)`, `body_pose (B*T, 69)`, `global_orient (B*T, 3)`, `transl (B*T, 3)`.
2. Wire the head into the cross-view model in `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`.
   - After `feat_pooled = feat.mean(dim=1)` (line 281), call the new head.
   - Return a dict of SMPL parameters alongside `pred_3d`, `weights`, `log_var`, `nll_loss`.

## Phase 2: Multi-task training loop (1 day)

3. Copy the trainer to `experiments/train_multi_task_shape_pose_mpiinf3dhp.py`.
   - Add CLI flags: `--shape_loss_weight`, `--pose_loss_weight`, `--reproj_weight`, `--shape_prior_weight`, `--bone_weight`, `--freeze_smpl_epochs`.
   - Load an frozen `smplx.SMPL(SMPL_MODEL_PATH, batch_size=B*T)` on the training device.
   - Loss:
     ```
     loss = L_3D
            + λ_SMPL * MSE(smpl_joints, gt_3d)
            + λ_reproj * weighted_reprojection(smpl_joints)
            + λ_shape * ||betas||^2
            + λ_bone * temporal_bone_length_consistency_loss(smpl_joints)
     ```
   - Freeze the SMPL head for the first `--freeze_smpl_epochs` epochs.

## Phase 3: IR plumbing & synthetic validation (1–2 days)

4. Update `motionflow_mv/ir/multiview_adapter.py` to populate `HumanMotionIR.pose` from the fusion output when SMPL keys are present.
5. Extend `experiments/generate_synthetic_multiview_dataset.py` to save GT `betas`, `body_pose`, `global_orient`, `transl`.
6. Create `experiments/eval_multi_task_shape_pose_synthetic.py`:
   - Load synthetic `.npz`.
   - Run fusion model.
   - Report 3D MPJPE, SMPL betas L2 error, body-pose axis-angle error, per-view reprojection error.

## Phase 4: Metrics & paper figures (0.5 day)

7. Add SMPL parameter metrics to `motionflow_mv/eval/metrics.py`:
   - `betas_error`, `body_pose_angle_error`, `smpl_reprojection_error`.
8. Add a smoke test in `tests/test_multi_task_shape_pose.py` that instantiates the head with random features and checks shapes/differentiability.

## Deliverables & exact file paths

- `motionflow_mv/fusion/smpl_shape_pose_head.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`
- `experiments/train_multi_task_shape_pose_mpiinf3dhp.py`
- `motionflow_mv/ir/multiview_adapter.py`
- `experiments/generate_synthetic_multiview_dataset.py`
- `experiments/eval_multi_task_shape_pose_synthetic.py`
- `motionflow_mv/eval/metrics.py`
- `tests/test_multi_task_shape_pose.py`

## Success criteria

- Synthetic smoke test completes in <5 min and converges to MPJPE < 10 mm.
- `betas` L2 error vs. GT < 0.5 on synthetic data.
- No regression on the existing MPI-INF-3DHP 3D MPJPE baseline.
