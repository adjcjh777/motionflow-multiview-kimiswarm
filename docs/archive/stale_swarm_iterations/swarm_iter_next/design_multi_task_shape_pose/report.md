# Multi-Task Shape & Pose Design for MotionFlow-MultiView

## Motivation

The current best model, `RayAttentionFusionModelTemporalResidual` (`motionflow_mv/fusion/ray_attention_temporal_residual_model.py`), is a pure 3D joint triangulation system. It outputs world-coordinate joints and per-view weights, but no parametric body. For ICRA/CVPR 2027, adding a **multi-task shape + pose branch** turns the fusion network into a true parametric body recovery pipeline, producing a valid `HumanMotionIR` with SMPL parameters directly from multi-view 2D evidence.

## Proposed Architecture

Keep the existing ray-aware attention, weighted DLT, and residual MLP intact — they already give a strong metric 3D estimate. Add a lightweight **SMPL shape/pose head** on top of the pooled temporal feature `f ∈ R^{B·T, J, d}`:

- **Shape head** `g_β(f)`: one shared `betas` vector per clip (temporally invariant).
- **Pose head** `g_θ(f)`: per-frame `body_pose` and `global_orient`.
- **Translation head** `g_t(f)`: per-frame `transl`.

The SMPL forward layer then yields 3D joints:

```
β   = g_β(f)                         (1, 10)
θ_t = g_θ(f_t)                       (B·T, 69)
ϕ_t = g_ϕ(f_t)                       (B·T, 3)
t_t = g_t(f_t)                       (B·T, 3)
J_SMPL_t = SMPL(β, θ_t, ϕ_t, t_t)    (B·T, 24, 3)
```

The final 3D pose used for benchmarking remains the residual-corrected triangulated output; the SMPL branch is trained as an auxiliary task and can be blended in future iterations.

## Implementation (task_04)

The task_04 prototype is implemented in two new files:

- `motionflow_mv/fusion/multi_task_shape_pose.py`
  - `MultiTaskShapePoseHead`: predicts shared `betas`, per-frame `body_pose`, `global_orient`, and `transl`. The head pools the concatenation of per-joint features and the raw triangulated 3D joints, then feeds a small MLP.
  - `MultiTaskShapePoseModel`: subclasses `RayAttentionFusionModelTemporalResidual` and intercepts the input to `residual_mlp` via a forward hook, feeding it to the shape/pose head. This avoids duplicating the parent forward pass and keeps the prototype self-contained.
  - `smplx` is an **optional** dependency; the head always emits parameter predictions, but it only runs the parametric body when `smplx` is installed and `SMPL_NEUTRAL.pkl` is supplied.

- `experiments/train_multitask_mpiinf3dhp.py`
  - Based on `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`.
  - Adds CLI flags: `--smpl_model_path`, `--shape_loss_weight`, `--pose_loss_weight`, `--shape_prior_weight`, `--freeze_smpl_head_epochs`.
  - Computes the multi-task loss `L = L_3D + λ_SMPL·L_SMPL_3D + λ_pose·||θ||² + λ_shape·||β||² + optional L_reproj + optional L_bone`.
  - Supports freezing the SMPL head for the first `N` epochs for warm-up stability.

## Multi-Task Loss

```
L = L_3D + λ_SMPL·L_SMPL_3D + λ_reproj·L_reproj + λ_shape·||β||² + λ_bone·L_bone
```

where `L_3D` is the existing 3D MSE on the triangulated/residual output, `L_SMPL_3D` is MSE between the first 17 SMPL joints and GT, and `L_reproj` enforces multi-view consistency via the predicted per-view weights. The shape prior `||β||²` and the bone-length consistency loss regularize the parametric body.

## Key Design Decisions

1. **Keep triangulation as the geometric anchor.** The SMPL head refines a camera-consistent 3D estimate, so metric scale is preserved without end-to-end pose regression drift.
2. **Shared clip-level shape.** A single `betas` vector per sequence is more stable than per-frame shape and matches the existing offline fitter (`experiments/fit_smpl_multiview.py`).
3. **Warm-up the SMPL head frozen.** The trainer supports freezing the SMPL head for the first `N` epochs, avoiding gradient instability through the parametric layer.
4. **Optional `smplx` dependency.** The module is importable and trainable without `smplx`; only the SMPL forward and 3D SMPL losses are skipped. This keeps continuous integration simple.
5. **Reuse the IR.** `motionflow_mv/ir/human_motion_ir.py` already stores `betas`, `body_pose`, `global_orient`, and `transl`, so the output can be written back without schema changes.

## Smoke Test & Validation

Run the built-in smoke test (no dataset required):

```bash
conda run -n mf python -m motionflow_mv.fusion.multi_task_shape_pose
```

Run a one-epoch sanity check on a synthetic `.npz` (generate one with the
required keys ``points_2d, confidences, joints_3d, camera_K, camera_R,
camera_t``):

```bash
conda run -n mf python experiments/train_multitask_mpiinf3dhp.py \
    --train tmp/synthetic_mpi.npz --val tmp/synthetic_mpi.npz \
    --clip_len 9 --d 32 --residual_hidden 64 --batch_size 2 \
    --train_samples 4 --epochs 1 --val_stride 10 \
    --output tmp/multitask_smoke.pth \
    --smpl_model_path data/smpl/SMPL_NEUTRAL.pkl
```

## References

- `motionflow_mv/fusion/multi_task_shape_pose.py`
- `experiments/train_multitask_mpiinf3dhp.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- `experiments/fit_smpl_multiview.py`
- `motionflow_mv/ir/human_motion_ir.py`
- `motionflow_mv/ir/multiview_adapter.py`
- `experiments/train_utils.py`
