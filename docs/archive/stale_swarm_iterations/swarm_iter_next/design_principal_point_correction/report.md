# Design Report: Principal-Point Correction Layer

## Motivation

Calibrated multi-view pipelines triangulate 2D keypoints using fixed intrinsic
calibration matrices `K`.  In practice, the principal point `(cx, cy)` is often
the least reliable part of `K`: it drifts across recordings, is hard to estimate
from checkerboard calibration when the subject is off-center, and is sometimes
simply set to the image center.  A small principal-point error biases the back-
projected camera rays and therefore the triangulated 3D pose.

This task adds a lightweight, learned principal-point correction layer that
adjusts `K` on-the-fly before triangulation.  Because the correction is bounded
and initialised near zero, it preserves the strong baseline behaviour of the
existing `RayAttentionFusionModelTemporalResidual` and only learns to fix small
calibration drift when it is useful.

## Approach

### 1. `PrincipalPointCorrection` (`motionflow_mv/fusion/principal_point_correction.py`)

A standalone `nn.Module` that predicts a per-view 2-D offset

```
Δ = tanh(MLP(pool(feat))) * max_offset   # (N, V, 2)
K_corrected[..., 0, 2] += Δ[..., 0]
K_corrected[..., 1, 2] += Δ[..., 1]
```

* **Input**: per-view per-joint features (`N, V, J, d`) plus an optional
  confidence/weight map for weighted pooling, or raw 2-D observations (`N, V, J, 3`)
  as a fallback.
* **Output**: corrected intrinsics (`N, V, 3, 3`) and the predicted offsets.
* **Bounds**: `tanh` squashing keeps the correction in `[-max_offset, max_offset]`
  pixels (default ±20 px).  At init the MLP outputs are near zero, so the layer
  is transparent.

### 2. New model file (`motionflow_mv/fusion/ray_attention_temporal_residual_principal_point_model.py`)

Subclasses the current best residual model and inserts the correction layer
between the temporal feature extractor and the triangulation step.

```
feat  ──> PrincipalPointCorrection ──> K_corrected
weights ────────────────────────────────────┘
                                            
                                        triangulate(P = K_corrected [R|t])
```

Key points:
* The correction is predicted from the **temporal per-view features** pooled
  over joints, so it can adapt to the specific frame/rig.
* The existing view/joint/temporal attention, weight head, and residual head are
  reused unchanged.
* Triangulation now uses the corrected intrinsics; the residual head sees the
  same corrected geometry.

## Files created

| Path | Purpose |
|------|---------|
| `motionflow_mv/fusion/principal_point_correction.py` | Reusable `PrincipalPointCorrection` layer. |
| `motionflow_mv/fusion/ray_attention_temporal_residual_principal_point_model.py` | New residual model with the layer integrated. |
| `docs/swarm_iter_next/design_principal_point_correction/report.md` | This report. |

## How to test / validate

Quick smoke tests (CPU, no real training):

```bash
wsl bash -c "cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm && .venv/bin/python motionflow_mv/fusion/principal_point_correction.py"
wsl bash -c "cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm && .venv/bin/python motionflow_mv/fusion/ray_attention_temporal_residual_principal_point_model.py"
```

The scripts check:
* Correct output shapes for `K_corrected` and predicted offsets.
* The predicted offsets are bounded by `max_offset`.
* The full model forward/backward pass works and gradients reach the new layer.

For a real impact check, train a small MPI-INF-3DHP model with a synthetic
principal-point perturbation (e.g. add `U(-10, 10)` px to `cx, cy` at training
time) and compare with/without the correction layer.  The design should reduce
MPJPE when calibration is noisy.

## Expected impact

* **Calibration robustness**: Small `cx, cy` errors are explicitly corrected,
  which is especially helpful when moving across datasets with different camera
  calibration pipelines.
* **Low risk**: bounded correction + near-zero init means the model falls back to
  the original residual model if the correction is not needed.
* **Metric improvement**: on MPI-INF-3DHP/H36M, correcting even a few pixels of
  principal-point drift can improve triangulation by a fraction of a millimetre.
  On the existing 10.46 mm MPJPE baseline, a 0.2–0.5 mm improvement is plausible
  if calibration noise is present.

## Implementation updates

* **Ray consistency fixed**: `RayAttentionFusionModelTemporalResidualPrincipalPoint`
  now computes ray embeddings with the corrected intrinsics `K_corrected`, so both
  the attention features and the triangulation step use the same geometry.
* **Explicit principal-point offset supervision**: the training script uses
  `perturb_cameras_with_delta` to recover the true per-view `(dx, dy)` applied
  during augmentation, and adds a weighted MSE loss
  `λ * ||Δ_pred − Δ_true||²` to teach the correction layer directly.

## Open blockers / next steps

* **Validation**: run the fast small ablation with `--pp_loss_weight=1.0` on the
  local RTX 4090 and verify that principal-point errors are no longer catastrophic.
* **Per-frame vs. global correction**: the current layer predicts a per-frame,
  per-view correction.  If the principal point is known to be constant for a whole
  sequence, adding a temporal smoothing constraint could improve stability.
* **Checkpoint compatibility**: the new model has extra parameters, so existing
  `RayAttentionFusionModelTemporalResidual` checkpoints cannot be loaded directly.
  A small migration script that copies compatible keys is useful if warm-starting
  from the current best checkpoint.
