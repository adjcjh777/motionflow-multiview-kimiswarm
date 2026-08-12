# Occlusion and partial visibility handling

## Problem statement

The current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, handles occlusion only implicitly through training-time dropout. While a visibility-gated variant (`CrossviewResidualVisibilityV2`) and an occlusion augmentation module (`motionflow_mv/data/occlusion_aug.py`) already exist, there is no CPU-only smoke harness that verifies the visibility head actually responds to synthetic occlusion, nor a minimal launcher that can queue the next real GPU experiment without touching the currently running curriculum job. We therefore need a focused, low-risk next step that tests the visibility-gated architecture under controlled occlusion and produces a clear recipe for the full GPU evaluation.

## Simplest concrete next step

Create and run a CPU-only smoke script that:

1. Generates a tiny synthetic 4-view, 17-joint dataset.
2. Trains a small `CrossviewResidualVisibilityV2` for 5 epochs with view/joint dropout augmentation and a BCE visibility loss.
3. Evaluates the trained model under three conditions: clean, 30% random view dropout, and 30% random joint dropout.
4. Reports MPJPE and per-view/per-joint binary visibility accuracy.

This validates the model, the augmentation module, and the evaluation harness without using a GPU or modifying any existing training runner.

## Files to touch

- `docs/swarm_iter7/occlusion_partial_visibility_handling.md` — this report.
- `experiments/eval_visibility_v2_occlusion_robustness_smoke.py` — new CPU-only smoke/evaluation script (created).
- `motionflow_mv/data/occlusion_aug.py` — no changes; reused by the synthetic dataset for reproducible masking.
- `motionflow_mv/models/crossview_residual_visibility_v2.py` — no changes; the script instantiates this existing model.

## Rough diff / sketch

```python
# New file: experiments/eval_visibility_v2_occlusion_robustness_smoke.py
# High-level flow (full file is in repo)

dataset = SyntheticOcclusionDataset(
    K, R, t, n_frames=120, clip_len=9,
    view_rate=0.2, joint_rate=0.2
)
model = CrossviewResidualVisibilityV2(
    j=17, d=32, n_views=4, n_st_layers=1,
    residual_hidden=64, visibility_hidden=32,
    principal_point_max_offset=0.0, focal_max_scale=0.0,
)
# Train 5 epochs with MSE + 0.1 * BCE(visibility, confidences > 0)
# Evaluate on clean / view_drop_30 / joint_drop_30
```

## Expected success metric

- Smoke completes on CPU in under 2 minutes.
- Clean synthetic MPJPE < 20 mm.
- 30% view/joint drop MPJPE degrades by < 25% relative to clean.
- Visibility accuracy ≥ 90% on synthetic occlusion labels.

## Resource requirement

CPU-only, no GPU, no long-running training. Safe to run while the WSL RTX 4090 is busy with the cross-view PP curriculum.

## Command and result

Run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/eval_visibility_v2_occlusion_robustness_smoke.py
```

Output:

```text
Device: cpu
Model: n_views=4, j=17, clip_len=9, d=32, params=56567
Training with view_rate=0.2, joint_rate=0.2
Epoch 1: train_loss=0.010690
Epoch 2: train_loss=0.000234
Epoch 3: train_loss=0.000115
Epoch 4: train_loss=0.000080
Epoch 5: train_loss=0.000060

=== Occlusion robustness (CPU smoke) ===
  clean           | MPJPE    4.14 mm | vis_acc 100.0%
  view_drop_30    | MPJPE    4.56 mm | vis_acc 100.0%
  joint_drop_30   | MPJPE    4.40 mm | vis_acc 100.0%

Saved checkpoint to outputs/visibility_v2_occlusion_smoke.pth
```

The smoke test passes all thresholds: clean MPJPE is 4.14 mm, view-drop degradation is ~10%, joint-drop degradation is ~6%, and visibility accuracy is 100% on the synthetic labels.

## Next GPU step (do not run now)

Once the RTX 4090 is free, queue the full visibility-gated training:

```bash
bash scripts/run_crossview_residual_visibility_v2_wsl.sh
```

Then evaluate the resulting checkpoint with the real MPI-INF-3DHP robustness harness:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/eval_occlusion_robustness.py \
    --checkpoint outputs/ray_attention_temporal_crossview_residual_visibility_v2_mpiinf3dhp.pth \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz
```

Target for the full run: clean MPI-INF-3DHP MPJPE ≤ 9.6 mm and ≥ 10% relative improvement at 30% random joint occlusion compared with the base PP checkpoint.
