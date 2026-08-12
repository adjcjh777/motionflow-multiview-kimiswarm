# Viewpoint-Equivariant Fusion: Enforce Equivariance to Camera Rotation

**Author:** iter14 agent (viewpoint-equivariant fusion)  
**Anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Date:** 2026-08-06

---

## 1. Problem

The current cross-view fusion model is trained on a single world-coordinate frame, so its 3-D predictions are not guaranteed to transform consistently when the entire multi-camera rig is rotated around the subject, leaving robustness to global viewpoint changes on an ad-hoc data-augmentation basis.

## 2. Hypothesis

If we explicitly constrain the network so that a global SO(3) rotation of all camera extrinsics (and the corresponding 3-D labels) rotates the predicted 3-D pose by the same amount, the model will retain the anchor’s clean accuracy while becoming more stable under calibration/rotation noise and more transferable to novel camera rigs.

## 3. Method

We enforce **SO(3) viewpoint equivariance** through a combination of on-the-fly rotation augmentation and an explicit consistency loss, without touching the core model graph.

### 3.1 New files to create

| File | Purpose |
|---|---|
| `motionflow_mv/losses/equivariance_loss.py` | `rotation_equivariance_loss` and a small `random_so3_matrix` helper. |
| `experiments/train_viewpoint_equivariant_residual_principal_point_mpiinf3dhp.py` | Fork of the existing PP trainer that injects random global rotations and the equivariance loss. |
| `experiments/eval_viewpoint_equivariance.py` | Measures how well a checkpoint obeys `pred(rotated rig) ≈ rotate(pred)` on a validation clip. |
| `tests/test_viewpoint_equivariance.py` | Unit test: synthetic 4-view rig, random SO(3), consistency error < 1 mm. |

### 3.2 Exact change in training

In each training batch, with probability `p=0.5` sample a random `Q ∈ SO(3)` (same matrix for the whole batch for simplicity) and build an rotated copy of the batch:

```python
# Q ~ SO(3), shape (3, 3)
R_rot = R @ Q.T          # world frame is rotated by Q
y_rot = y @ Q.T          # ground-truth skeleton in the rotated frame
t_rot = t                # translation is invariant under pure rotation
K_rot = K                # intrinsics are unchanged
```

The model is forwarded twice: once on `(x, K, R, t)` and once on `(x, K_rot, R_rot, t)`. The supervised loss is computed on **both** outputs against their corresponding labels, plus an explicit equivariance consistency term:

```python
# pred_rot already lives in the rotated world frame;
# pred_orig rotated by Q should match it.
equiv_loss = mse(pred_rot, pred_orig @ Q.T)
loss = mse(pred_orig, y) + mse(pred_rot, y_rot) + lambda_equiv * equiv_loss
```

`lambda_equiv` is a small scalar (start at `0.1`); if it over-regularises, it is reduced in the fallback.

### 3.3 Where the existing code is touched

No existing module is modified. The new trainer is a stand-alone fork of `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` and reuses:

- `motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model.RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
- `motionflow_mv.calibration.perturb.perturb_cameras_with_delta`
- `motionflow_mv.losses.reprojection_loss`
- `motionflow_mv.losses.velocity_loss`
- the existing `TemporalClipDataset` / `RandomClipDataset` collate logic

The only new dependency is `motionflow_mv.losses.equivariance_loss`, which is imported by the new trainer.

## 4. Smoke-Test Plan

Run a **5-epoch** smoke on MPI-INF-3DHP S1, using the same small configuration that already passed the factorized PP smoke.

```bash
conda run -n mf python experiments/train_viewpoint_equivariant_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val   data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --epochs 5 --train_samples 500 --batch_size 8 \
  --pp_loss_weight 0.1 --cam_aug_pp 2.0 --cam_aug_focal 0.01 \
  --equiv_loss_weight 0.1 --equiv_aug_prob 0.5 \
  --output outputs/viewpoint_equivariant_smoke.pth
```

**Pass/fail criteria:**

- **Pass:** val MPJPE ≤ **9.8 mm** and PA-MPJPE ≤ **5.8 mm** on the 500-sample smoke (within ~0.5 mm of the anchor).
- **Pass:** no NaNs or gradient explosions; `max_grad_norm` stays finite.
- **Pass:** equivariance residual on 50 validation clips ≤ **5 mm** after 5 epochs.
- **Fail:** val MPJPE > **10.5 mm**, any NaN, or equivariance residual > **10 mm**.

## 5. Evaluation Plan

1. **Clean metrics** — run the existing evaluator to confirm the anchor-level numbers are preserved:
   ```bash
   python experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
     --checkpoint outputs/viewpoint_equivariant_smoke.pth \
     --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
     --out_json outputs/viewpoint_equivariant_smoke_eval.json
   ```
   Report `mpjpe`, `pa_mpjpe`, `pck@50/100/150`, `pck_auc` via `motionflow_mv.eval.metrics.compute_all_metrics`.

2. **Equivariance residual** — run the new script:
   ```bash
   python experiments/eval_viewpoint_equivariance.py \
     --checkpoint outputs/viewpoint_equivariant_smoke.pth \
     --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
     --n_rotations 5 --out_json outputs/viewpoint_equivariance_residual.json
   ```
   Report `equiv_error = mean || pred(R@Q^T) - pred(R) @ Q^T ||` in millimetres.

3. **Calibration-robustness spot check** — compare the `rot_0.5_deg` / `rot_1.0_deg` rows from `experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` against the anchor. Target: ≥10 % relative improvement on `rot_1.0_deg` without clean regression.

## 6. Estimated GPU/CPU Cost on RTX 4090

- **Training smoke:** 500 random clips × 5 epochs ≈ **20 min** on a single RTX 4090.
- **Memory:** peak GPU memory ≈ **6 GB** with `d=32`, `batch_size=8`, `clip_len=13`.
- **Evaluation:** clean + equivariance check < **2 min** CPU, mostly data loading.

Total one-shot cost: **~25 min** RTX 4090 time.

## 7. Risks & Fallback

| Risk | Mitigation / fallback |
|---|---|
| The explicit consistency loss over-regularises the pose and hurts clean MPJPE. | Drop `equiv_loss_weight` to `0.01` or disable the explicit term and keep only the SO(3) data augmentation (second forward + rotated labels). |
| View positional embeddings / learned camera embeddings are not rotation-covariant, so the model only learns approximate equivariance. | If residual stays >10 mm, pivot to a canonical-frame architecture: rotate per-view features into a body-centered frame before attention and unrotate the triangulated output. |
| Rotating `R` while keeping `K` fixed may confuse the principal-point head because the image-plane axes are unchanged but the world axes move. | Freeze the PP/focal head for the first epoch of the smoke and keep `cam_aug_pp` / `cam_aug_focal` low. |
| Smoke is too slow on RTX 4090. | Reduce `train_samples` to 200 and `clip_len` to 9; the pass/fail thresholds are scaled proportionally. |

---

## Summary

This proposal adds **no new model parameters**; it enforces SO(3) viewpoint equivariance by rotating the entire camera rig on-the-fly and penalising inconsistent outputs. If the smoke shows clean accuracy within 0.5 mm of the anchor and an equivariance residual below 5 mm, the method is worth a full-length run and integration into the robustness matrix.
