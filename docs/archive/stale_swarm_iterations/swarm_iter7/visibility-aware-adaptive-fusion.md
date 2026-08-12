# Visibility-aware Adaptive Fusion

## Problem statement

The current best model (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) fuses every available view when it triangulates 3D joints. When a view is occluded, its corrupted 2D keypoint can still leak into the DLT solution through the learned view weights, hurting robustness. Adding an explicit per-view/per-joint visibility head that soft-gates the DLT weights — with a fallback guard to avoid degenerate triangulation — should make the system substantially more robust to missing or corrupted views, and is a natural complement to the variable-view and view-dropout work already in progress.

## Simplest concrete next step

Run a CPU-only synthetic occlusion smoke test of the existing visibility-gated PP model to verify that (1) the visibility head receives gradients, (2) it learns to identify corrupted views, and (3) gating the DLT weights with that mask lowers 3D error. Once the RTX 4090 GPU queue is free, launch the full MPI-INF-3DHP training via `scripts/run_crossview_pp_visibility_wsl.sh` (warm-started from the best PP checkpoint, with `view_dropout_rate=0.2` and `visibility_loss_weight=0.1`).

## Files to touch / sketch

- **New CPU smoke test:** `docs/swarm_iter7/smoke_visibility_adaptive_fusion.py`
  - Builds a 4-view synthetic rig, corrupts 1–2 views per joint, and trains `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility` for 120 steps.
  - Loss: `MSE(pred_3d, target_3d) + 0.1 * BCE(predicted_visibility, true_visibility)`.
  - Asserts visibility mask accuracy > 0.75 and a reduction in 3D error after training.
- **GPU launcher (already exists, do not modify):** `scripts/run_crossview_pp_visibility_wsl.sh`
- **Training script (already exists, do not modify):** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_visibility_mpiinf3dhp.py`
- **Model (already exists, do not modify):** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_visibility_model.py`

## Expected success metric

- **CPU smoke test:** visibility mask accuracy > 0.75 and 3D error after training lower than before training on held-out occluded views.
- **GPU training:** MPI-INF-3DHP clean MPJPE ≤ 9.6 mm; ≥ 10% relative error reduction at 30% synthetic occlusion compared to the baseline PP model.

## Resource requirements

- CPU-only for the smoke test; can be run now without touching the running GPU job.
- GPU only for the full MPI-INF-3DHP training (queued until the cross-view PP curriculum finishes).

## Run command and result

```bash
KMP_DUPLICATE_LIB_OK=TRUE python docs/swarm_iter7/smoke_visibility_adaptive_fusion.py
```

Output:

```text
Before training: err=432.79mm, vis_acc=0.500
step  30: 3d_loss=0.0020 vis_loss=25.0000
step  60: 3d_loss=0.0006 vis_loss=15.1884
step  90: 3d_loss=0.0012 vis_loss=2.3108
step 120: 3d_loss=0.0006 vis_loss=2.4478
After training: err=74.61mm, vis_acc=0.863
smoke test passed
```

The visibility head quickly learns to distinguish corrupted views (visibility accuracy rises from 0.50 to 0.863) and the gating mechanism reduces the synthetic 3D reconstruction error. This validates the plumbing for the visibility-aware adaptive fusion direction; the next step is the GPU training run once the queue is free.
