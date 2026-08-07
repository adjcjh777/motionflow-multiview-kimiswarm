# v10 Design: Calibration-Perturbation Curriculum for Stable v9+ Training

## 1. Problem in the current v7/v8/v9 pipeline

`experiments/train_omniview_fusion_v5_webbridge_multi.py` applies a single,
pre-defined calibration curriculum through
`apply_calibration_perturbation()` (lines 331–350):

- All perturbation modes (rotation, translation, focal length, principal point)
  ramp on a fixed epoch schedule in `motionflow_mv/calibration/camera_perturbation_curriculum.py`.
- The schedule is **agnostic to the rest of the loss mix**, especially the new
  v9 losses: 2D reprojection loss (`_reprojection_loss`, lines 443–477),
  Procrustes-aligned loss, and the monotonic multi-view loss (lines 661–675).

Why this hurts v9 specifically:

1. **Reprojection loss explodes under simultaneous extrinsic/intrinsic noise.**
   Reprojection MSE is measured in squared pixels. If extrinsic and intrinsic
   perturbations are both large early on, the 2D points and the perturbed
   cameras are geometrically inconsistent, so the reprojection term dominates and
   destabilizes the whole loss surface (observed: loss ~3000 at step 50–100).

2. **Robust DLT reweight in v8 over-reacts to early extrinsic noise.**
   In `omniview_fusion_v5.py` (lines 516–533) the robust weights `rho` are
   detached and clamped, but large early calibration errors can still push the
   weight distribution toward near-uniform or near-zero values before the
  2D/camera geometry head has learned anything stable.

3. **The monotonic loss is evaluated on subsets sampled with perturbed cameras.**
   When the camera perturbation is still large and the model is immature, the
   monotonic ranking signal is mostly noise, adding gradient variance.

4. **The current curriculum has no rollback / stability guard.**
   If a given epoch’s perturbation level is too hard for the current loss mix,
   training must live with it; there is no mechanism to dial the perturbation
   back until the model recovers.

In short: v7/v8 worked because the main loss was a clean 3-D MSE + epipolar
term; v9 added geometry-sensitive losses that must be introduced with a
calibration curriculum matched to their sensitivity.

## 2. Concrete v10 design: “Loss-gated, decoupled calibration curriculum”

### 2.1 High-level idea

Split the calibration curriculum into **independent, loss-gated stages**:

1. **Clean warm-up** (first `W` epochs): no calibration perturbation. Train the
   model on clean mixed-dataset, variable-view samples so the DLT/robust
   reweight path stabilizes.
2. **Extrinsics-only ramp** (next `E` epochs): slowly increase rotation and
   translation noise. Keep intrinsics clean so the reprojection loss still sees
   a consistent camera model.
3. **Intrinsics-only ramp** (next `I` epochs): add focal-length and principal-point
   noise only after extrinsics have reached their target and the reprojection loss
   has stayed bounded.
4. **Optional light anneal** (last `A` epochs): reduce all perturbation stds by
   20–30 % to recover a small amount of clean-data accuracy.

Gate the transitions with a **loss-guard**:

- Track the ratio `reproj_loss / mse_loss` per epoch.
- If the ratio exceeds a threshold `τ` for two consecutive steps, **hold the
  curriculum at the current level** and reduce the new perturbation step size by
  50 % until the ratio drops below `τ`.
- If the reprojection loss is enabled, delay its activation until the
  extrinsics stage is finished.

### 2.2 Minimal code change

Add a new schedule name and a small helper in
`motionflow_mv/calibration/camera_perturbation_curriculum.py`:

```python
# New schedule: “reproj_gated_curriculum”
def loss_gated_camera_schedule(epoch, *, base_schedule_fn, reproj_ratio, tau=3.0):
    stds = base_schedule_fn(epoch)
    if reproj_ratio > tau:
        stds = {k: v * 0.5 for k, v in stds.items()}   # rollback
    return stds
```

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, extend the
argument parser with three flags only:

- `--cam_aug_loss_gate_ratio τ` (default 3.0)
- `--cam_aug_stage_epochs W,E,I,A` (default `2,5,5,0`)
- `--reproj_loss_delay_epochs D` (default 3)

Then replace the single call to `apply_calibration_perturbation()` with a
small state machine:

```python
if epoch < W:
    stds = {k: 0.0 for k in (...)}           # clean warm-up
elif W <= epoch < W+E:
    stds = ramp_extrinsics_only(epoch - W)   # rot/trans ramp
elif W+E <= epoch < W+E+I:
    stds = ramp_intrinsics_only(epoch - W - E)  # focal/pp ramp
else:
    stds = full_schedule(epoch)

# Loss-gated rollback
if reproj_ratio > args.cam_aug_loss_gate_ratio:
    stds = {k: v * 0.5 for k, v in stds.items()}

K_aug, R_aug, t_aug = perturb_cameras(K, R, t, **stds)
```

Disable the reprojection and monotonic losses until
`epoch >= args.reproj_loss_delay_epochs`, regardless of their weight flags.
Keep the PA loss active from the start because it only operates on -D pose
shape and is insensitive to camera noise.

## 3. Validation plan

### 3.1 Smoke test (~5 minutes)

Run the synthetic CPU/GPU smoke path built into the training script:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --cam_aug_schedule extended_curriculum \
  --reproj_loss_delay_epochs 1
```

Checks:
- Training completes one epoch without NaN/Inf or loss explosion.
- `apply_calibration_perturbation()` returns finite K, R, t.
- The reprojection loss is zero until the delayed epoch and finite afterward.

### 3.2 Small fast run (~1–2 hours on A800-D)

Use a small mixed-dataset smoke subset (same WebBridge H36M+MPI mixed loader as
v7/v8) and train for 10 epochs with the v8 flag set plus the new curriculum:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader --mixed_manifest configs/splits/webbridge_h36m_mpi_smoke.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight \
  --use_domain_embedding \
  --cam_aug_schedule extended_curriculum \
  --cam_aug_rot 0.5 --cam_aug_trans 0.005 \
  --cam_aug_focal 0.01 --cam_aug_pp 2.0 \
  --cam_aug_ramp_epochs 5 --cam_aug_intrinsics_ramp_epochs 3 \
  --cam_aug_loss_gate_ratio 3.0 --reproj_loss_delay_epochs 3 \
  --reproj_loss_weight 0.1 --pa_loss_weight 0.1 \
  --monotonic_loss_weight 0.01 \
  --epochs 10 --batch_size 8 --clip_len 13 \
  --output outputs/v10_reproj_gated_curriculum_smoke.pth
```

Success criteria:
- Step-50 total loss < 100 (vs. v9 ~3000).
- Average reprojection loss / MSE loss < 5 after epoch 3.
- Val MPJPE comparable or better than the v8 smoke baseline.

### 3.3 Full run (~1–2 days on A800-D)

Same command on the full WebBridge H36M+MPI mixed manifest, 30 epochs. Track:

- Clean validation MPJPE.
- Robustness matrix: `rot_0.5°`, `focal_1%`, `pp_10px`, `trans_5mm`.
- Per-epoch reprojection/MSE ratio and histogram of robust DLT weights.

## 4. Expected impact

| Metric | Expected effect |
|--------|-----------------|
| **Training stability** | v9-style loss explosion eliminated; step-50 loss stays below ~100. |
| **Clean MPJPE** | Neutral to +1 %; the clean warm-up preserves v7/v8 accuracy. |
| **Robustness (rot_0.5°, focal_1%)** | 2–5 mm improvement over flat perturbation because extrinsics and intrinsics are introduced separately and the robust reweight path has time to learn meaningful per-view precision. |
| **Monotonic loss utility** | Becomes useful after epoch 3 instead of being swamped by camera noise. |
| **Triangulation weight interpretability** | `motionflow_mv/fusion/attention_entropy_loss.py` (used optionally via `use_entropy_regularization`) benefits from cleaner early weights. |

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Slower robustness adaptation** because perturbations are delayed. | Keep the final stage at full strength and optionally add a short “burn-in” at the end with `rot_std`/`focal_std` maxed out. |
| **Robust DLT still over-downweights views** under large extrinsics. | The v8 clamps on `rho` and weight floors already exist; add a histogram check of `weights` in the full run. |
| **New hyperparameters are hard to tune.** | Only three new flags are introduced; defaults are chosen to match the existing extended curriculum except for the gating. |
| **Reprojection loss delay starves multi-view geometry learning.** | Keep the epipolar consistency loss (`epipolar_loss_weight`) and the camera-conditioned embedding active during the clean warm-up. |
| **Mixed-dataset domain shift interacts with camera noise.** | `use_domain_embedding` stays on; domain ID is still passed to `omniview_fusion_v5.py` (line 416). |

## 6. Files touched

- `motionflow_mv/calibration/camera_perturbation_curriculum.py` — add the loss-gated schedule helper.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — stage the curriculum, add the three new flags, and gate reprojection/monotonic loss activation.
- `tests/test_camera_perturbation_curriculum.py` — add a test for the new schedule and the loss-gate rollback.
- `docs/swarm_iter_next/design_calibration_perturbation_curriculum_v10.md` — this document.

## 7. Roll-out order

1. Implement the schedule helper and the three CLI flags.
2. Run the smoke test.
3. Run the 10-epoch fast run; compare to v8 smoke baseline.
4. If stable, launch the 30-epoch full run in parallel with a v9 control run
   (reprojection loss from epoch 0, flat perturbation).
5. Evaluate robustness matrix and decide whether to make this the default
   curriculum for v10.
