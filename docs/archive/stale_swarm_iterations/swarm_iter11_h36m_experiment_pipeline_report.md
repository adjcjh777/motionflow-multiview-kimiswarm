# Iter11+ H36M Experiment Pipeline Roadmap

## Executive Summary

The MotionFlow-MultiView codebase now has a strong advanced fusion model (`ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`) that combines spatio-temporal attention, uncertainty-weighted DLT, differentiable Gauss-Newton triangulation, and residual refinement.  It has been trained on MPI-INF-3DHP clips, but **Human3.6M (H36M) has no equivalent trainer or evaluation script yet**.  This report proposes a concrete, immediately implementable H36M experiment pipeline: port the full model to H36M, switch to the canonical cross-subject protocol, add robust augmentation/curriculum, add reprojection and skeleton losses, and run a standardized evaluation/ablation suite.

## Current State

- **Data**: `data/h36m_hf/` and `data/webbridge/h36m/` contain per-subject/action `.npz` files in **millimeters** (`points_2d` in px, `joints_3d` in mm, `camera_t` in mm).  `data/webbridge/h36m/` already has 103 files covering subjects 1–16.
- **Trainers**: H36M scripts (`train_ray_attention_v1_h36m.py`, `v3_h36m.py`, `v4_h36m.py`) train per-frame with simple train/val splits.  None use temporal clips, uncertainty, Gauss-Newton refinement, or the cross-view transformer.
- **Best model elsewhere**: The cross-view residual model reaches ~11.17 mm MPJPE on MPI-INF-3DHP validation.  A fast temporal-residual run hit 47.54 mm, attributed to limited data/epochs.
- **Gap**: There is no `experiments/train_*_h36m.py` for the full advanced model, no canonical H36M protocol, and no systematic robustness evaluation.

## Proposed Improvements

### 1. Port the Full Advanced Model to H36M

Create `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_h36m.py` by adapting the MPI-INF-3DHP script (`train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`).

Reuse the existing `TemporalClipDataset` / `RandomClipDataset` clip loaders, but keep H36M units in **mm** (do not divide by 1000).  The advanced model is unit-agnostic as long as `camera_t` and `joints_3d` share the same scale.

Suggested default hyperparameters for a first run:

| Parameter | Value |
|---|---|
| `clip_len` | 13 (try 27 for long actions) |
| `d` | 64 |
| `n_st_layers` | 2 |
| `residual_hidden` | 128 |
| `gn_iters` | 3 |
| `batch_size` | 8 |
| `lr` | 1e-3 |
| `epochs` | 50 |
| `uncertainty_weight` | 0.1 |
| `reproj_weight` | 0.01 |

### 2. Adopt the Canonical Cross-Subject Protocol

Replace the random 90/10 split with the literature-standard H36M split so results are comparable and publishable:

- **Train**: subjects 1, 5, 6, 7, 8
- **Validation**: subject 9
- **Test**: subject 11

Point the trainer at the `data/webbridge/h36m/` files and concatenate them with `torch.utils.data.ConcatDataset` (as is already done for MPI-INF-3DHP).

### 3. Robustness Curriculum + Auxiliary Losses

Port the `RobustnessCurriculum` from `experiments/train_ray_attention_v4_h36m.py` into the temporal trainer.  Ramping schedule (px/rate):

- 2D Gaussian noise: 0 → 5 px
- View dropout: 0 → 30%
- Sparse outliers: 0 → 5% at scale up to 100 px
- Warmup: 5 epochs

Add the auxiliary losses already available in the repo:

- **Reprojection loss** (`motionflow_mv.losses.reprojection_loss`) to enforce geometric consistency.
- **Bone-length loss** and **skeleton-consistency loss** (`experiments/train_utils.py`) for anatomical plausibility.

Proposed combined loss:

```python
pred, weights, log_var, nll = model(xb, K=K, R=R, t=t)
loss = criterion(pred, yb)
loss += args.reproj_weight   * reprojection_loss(pred, p2d, K, R, t, conf)
loss += args.bone_weight     * bone_length_loss(pred, yb, parents=H36M_17_PARENTS)
loss += args.consistency_weight * skeleton_consistency_loss(pred, ...)
loss += nll  # uncertainty NLL already weighted inside the model
```

### 4. Comprehensive Evaluation Script

Create `experiments/eval_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_h36m.py` that reports:

- MPJPE (mm)
- PA-MPJPE (mm)
- PCK@50/100/150 mm
- PCK AUC 0–150 mm
- Per-joint MPJPE/PA-MPJPE
- Per-action MPJPE
- Robustness grid: noise ∈ {0, 2, 5 px}, dropout ∈ {0, 0.2, 0.4}, outliers ∈ {0, 0.05}

Reuse `motionflow_mv.eval.metrics.compute_all_metrics` (already returns all these values).

### 5. Ablations

Run a short ablation matrix on the full H36M train/val split to isolate gains:

1. Full advanced model (temporal + cross-view + uncertainty + GN + residual)
2. No uncertainty (fixed DLT weights)
3. No Gauss-Newton refinement
4. No residual refinement
5. No temporal context (`clip_len=1`)
6. Baseline weighted DLT only

Track MPJPE, parameter count, and wall-clock training time for each.

### 6. Data-Quality Checks

- Verify that `data/webbridge/h36m/` camera ordering matches `camera_params.json` ordering.
- Optionally threshold low-confidence detections (`conf < 0.2 → 0`) before triangulation.
- If a direct H36M 3D GT source is available later, compare the triangulated targets against it; otherwise report that targets are DLT-reconstructed and note this as a limitation.

## Experiments to Run

| Experiment | Command / Script | Success Criterion |
|---|---|---|
| Full-model H36M train | `train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_h36m.py` | Lowest val MPJPE on S9 |
| Robustness evaluation | `eval_*_h36m.py --noise_levels 0 2 5 --dropout_rates 0 0.2 0.4` | Degradation Δ vs clean ≤ 20% at moderate corruption |
| Ablation study | `ablate_h36m_advanced_v1.py` (new) | Identify which component gives ≥1 mm gain |
| Cross-dataset check | Fine-tune S9→S11, eval on MPI-INF-3DHP | Relative transfer gap |

## Metrics to Track

- **Primary**: H36M val/test MPJPE (mm), PA-MPJPE (mm)
- **Secondary**: PCK@50/100/150 mm, AUC, per-joint errors, per-action errors
- **Robustness**: MPJPE under noise/dropout/outlier corruption
- **Training diagnostics**: loss components (MSE, reproj, bone, NLL), best epoch, convergence time, checkpoint size

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| H36M data is triangulated, not raw GT; camera/keypoint errors propagate | Report triangulation-based GT as a limitation; use confidence thresholding |
| Full model overfits on limited H36M multi-view clips | Use the cross-subject split, early stopping, and robustness curriculum |
| Unit mismatch (mm vs m) | Keep mm consistently; report MPJPE in mm; do not use the `_m.npz` variants for this pipeline |
| Compute/memory blow-up from `B×T×V×J` attention | Start with `clip_len=13`, `batch_size=8`; scale up if memory allows |
| Gauss-Newton instability during training | Use `gn_damping=1e-6` and gradient clipping (`max_norm=1.0`) |
| Gains from advanced model are smaller on H36M than on MPI-INF-3DHP | Ablations will show whether the extra capacity is justified |

## Recommended Next Steps

1. Create `train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_h36m.py` and validate one epoch on `s_01_acts_02`.
2. Convert the robustness curriculum into a shared helper (`experiments/augmentation.py`) so it can be reused across H36M trainers.
3. Run the full-model training on the canonical S1/S5/S6/S7/S8→S9 split.
4. Produce the evaluation report and the ablation matrix for the paper.
