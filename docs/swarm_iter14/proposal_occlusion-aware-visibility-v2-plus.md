# Occlusion-aware visibility v2+: visibility gating + occlusion-aware residual

## 1. Problem

The visibility-gated PP model (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility`) can mask out occluded views, but once a view is dropped the triangulated pose is still produced by the same static fusion network that assumes full visibility, so it has no learned way to *compensate* for the missing information by exploiting spatial relationships among visible joints.

## 2. Hypothesis

Adding a small, visibility-conditioned **occlusion-aware residual** that refines the triangulated 3-D pose from the remaining visible views will preserve the clean-anchor accuracy while reducing the error gap between full-visibility and occluded inputs.

## 3. Method

### 3.1 Architecture changes

Create a new model that extends `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility` and inserts an residual correction branch after the DLT triangulation.

- **New module to create:** `motionflow_mv/fusion/occlusion_aware_residual.py`
  - Implement `OcclusionAwareResidual(j, d, n_views, hidden=128)`.
  - Inputs:
    - `feat`: per-view spatio-temporal features `(B*T, V, J, d)`
    - `visibility`: effective visibility multipliers `(B*T, V, J)`
    - `pose_3d`: current triangulated 3-D joints `(B*T, J, 3)`
  - Computes a masked view-pooling of `feat` using `visibility` as weights, followed by two MLP layers that output a per-joint 3-D residual `delta_pose`.
  - Output: refined 3-D pose `pose_3d + delta_pose`.

- **New model to create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_visibility_occlusion_aware_model.py`
  - Define `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityOcclusionAware` subclassing `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility`.
  - In `forward`, after the parent produces the triangulated pose, pass `feat`, `effective_visibility`, and the pose through `OcclusionAwareResidual`.
  - Return the refined pose, visibility weights, and the raw residual for diagnostics.

### 3.2 Loss / objective changes

- Keep the existing 3-D MSE loss and the BCE visibility loss from visibility v2.
- Add a residual regularisation term: `residual_reg = 0.01 * ||delta_pose||_2` to keep the occlusion branch conservative.
- Optionally add a bone-length consistency loss on the refined output only when dropout is active (probability > 0):
  `bone_loss = mean(|bone_len(refined) - bone_len(ground_truth)|)`.
  Weight: `0.1` only for training batches where `view_dropout_rate > 0` or `joint_dropout_rate > 0`.

### 3.3 Data / augmentation changes

- Extend the augmentation helper in the trainer to inject both **view-level** and **joint-level** synthetic occlusion at training time.
  - `view_dropout_rate`: sample a random subset of views and zero their confidence.
  - `joint_dropout_rate`: sample a random subset of joints and zero their confidence across all views.
- These augmentations are already supported by `augment_clip` in `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`; the new trainer exposes both knobs and defaults to `view_dropout_rate=0.2`, `joint_dropout_rate=0.1`.

### 3.4 Exact files to create or modify

| Path | Action |
|------|--------|
| `motionflow_mv/fusion/occlusion_aware_residual.py` | Create new module |
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_visibility_occlusion_aware_model.py` | Create new model |
| `experiments/train_crossview_residual_visibility_v2_occlusion_mpiinf3dhp.py` | Create trainer (copy from `experiments/train_crossview_residual_visibility_v2_mpiinf3dhp.py` and add residual branch + joint dropout) |
| `scripts/run_crossview_residual_visibility_v2_occlusion_wsl.sh` | Create launcher script |
| `experiments/eval_visibility_v2_occlusion_aware_smoke.py` | Create 3–5 epoch smoke script |

### 3.5 Code sketch

```python
# motionflow_mv/fusion/occlusion_aware_residual.py
import torch
import torch.nn as nn


class OcclusionAwareResidual(nn.Module):
    def __init__(self, j: int = 17, d: int = 64, hidden: int = 128):
        super().__init__()
        self.j = j
        self.d = d
        self.mlp = nn.Sequential(
            nn.Linear(d + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, feat, visibility, pose_3d):
        # feat: (N, V, J, d), visibility: (N, V, J), pose_3d: (N, J, 3)
        masked = feat * visibility.unsqueeze(-1)  # zero-out occluded views
        pooled = masked.sum(dim=1) / (visibility.sum(dim=1, keepdim=True).clamp(min=1).transpose(-1, -2) + 1e-6)
        # pooled: (N, J, d)
        inputs = torch.cat([pooled, pose_3d], dim=-1)  # (N, J, d+3)
        delta = self.mlp(inputs)  # (N, J, 3)
        return pose_3d + delta, delta
```

## 4. Smoke-test plan

- **Script:** `experiments/eval_visibility_v2_occlusion_aware_smoke.py`
- **Sample:** 500 clips from MPI-INF-3DHP train / 100 clips validation (or reuse the synthetic loader if MPI is unavailable).
- **Model config for smoke:** `d=32`, `residual_hidden=64`, `visibility_hidden=32`, `occlusion_hidden=64`, `clip_len=13`, `batch_size=4`, `epochs=5`.
- **Pass / fail:**
  - Pass: training finishes without NaNs / crashes.
  - Pass: val MPJPE ≤ 10.0 mm (smoke threshold; clean full run target is ≤ 9.6 mm).
  - Pass: visibility prediction accuracy ≥ 0.75.
  - Pass: under 30 % view dropout, relative degradation vs. clean is ≤ 15 %.
  - Fail: any NaN, val MPJPE > 10.5 mm, or dropout degradation > 25 %.

## 5. Evaluation plan

Run on the same canonical MPI-INF-3DHP split used for the 9.32 mm anchor.

- **Metrics:** MPJPE, PA-MPJPE, PCK@50/100/150, AUC, per-joint visibility accuracy.
- **Scripts:**
  - `experiments/train_crossview_residual_visibility_v2_occlusion_mpiinf3dhp.py --epochs 30 --train ... --val ...`
  - `experiments/eval_visibility_v2_occlusion_aware_smoke.py` for quick validation.
  - Extend `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` to instantiate the new model and add `view_dropout` / `joint_dropout` severity sweeps.
- **Comparison:** baseline PP (9.32 mm), visibility v2, and the new visibility v2+ model.
- **Target:** clean MPJPE ≤ 9.6 mm (within 3 % of anchor) and at 30 % view dropout a ≥ 10 % relative improvement over the visibility v2 model.

## 6. Estimated GPU/CPU cost on RTX 4090

- **Smoke (5 epochs, 500 samples, d=32):** ~10–15 minutes on RTX 4090; CPU-only pre-processing may dominate. Memory < 4 GB.
- **Full run (30 epochs, full MPI-INF-3DHP):** ~4–6 hours on RTX 4090, comparable to the visibility v2 trainer. The occlusion-aware residual adds < 5 % parameters and negligible latency.

## 7. Risks & fallback

- **Risk:** The residual branch overfits to synthetic dropout and harms clean accuracy.
  - *Fallback:* Freeze the visibility head and train only the residual branch for the first 10 epochs, then unfreeze.
- **Risk:** Joint-level dropout during training destabilises the temporal residual.
  - *Fallback:* Disable joint dropout, keep only view dropout, and reduce dropout rate to 0.15.
- **Risk:** Visibility v2 training is already CPU-bound / slow.
  - *Fallback:* Run the smoke with `num_workers=0` and a cached pre-processed `.npz`; if still too slow, evaluate on the synthetic loader only.
- **Risk:** Residual correction provides no improvement over triangulation-only v2.
  - *Fallback:* Remove the residual branch and keep only the visibility gating as the anchor.
