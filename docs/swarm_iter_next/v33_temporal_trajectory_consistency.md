# v33 Proposal: Temporal Trajectory Consistency and Smoothness

**Direction:** `temporal_trajectory_consistency`  
**Date:** 2026-08-08  
**Author:** Kimi Code agent (swarm direction #20)  

## Problem statement and motivation

The current `OmniMultiViewFusionV5` pipeline optionally applies a v32 trajectory-consistency refiner (`TrajectoryConsistencyRefinerV32`) after the per-frame 3-D pose head. It is a single-scale 1-D temporal CNN over the flattened `(J*3)` pose vector with a scalar residual gate and a fixed MSE smoothness/drift loss. While it removes the broken v27/v29 test-time self-evolution path, it has three clear weaknesses:

1. **No confidence/visibility input.** The smoother treats all joints and frames equally, even when some joints are occluded or low-confidence.
2. **Single scale and scalar gate.** Every joint and motion regime uses the same smoothing kernel, which risks over-smoothing fast motion.
3. **Non-robust loss.** The MSE smoothness term is sensitive to spurious outliers (e.g., from outlier-view augmentation) and does not distinguish true high-frequency motion from noise.

The v33 direction proposes a **learned, confidence-aware, multi-scale temporal trajectory refiner** that can adapt per-joint smoothing strength, ignore low-confidence frames, and use a robust smoothness loss.

## Proposed architecture changes

### New module: `motionflow_mv/fusion/trajectory_consistency_v33.py`

Implement `TemporalTrajectoryConsistencyV33(nn.Module)`:

- **Inputs:**
  - `x`: predicted 3-D pose sequence, shape `(B, T, J, 3)`.
  - `conf`: optional per-joint confidence/visibility, shape `(B, T, J)`.
- **Per-joint multi-scale temporal convolutions:**
  - For each joint independently, apply parallel 1-D temporal convolutions with kernel sizes `{3, 5, 9}` (padding to keep length).
  - A learned per-scale weighting and a per-joint soft gate combine the residuals.
  - The final conv layer is zero-initialised and the gate is initialised to zero → the module is a no-op at the start of training.
- **Confidence-aware masking (optional):** When `conf` is provided, missing/low-confidence joints are excluded from the smoothness loss computation and their residual contribution is down-weighted.
- **Lightweight temporal self-attention branch (optional, ablatable):** A single-layer transformer over `(B, T, J*3)` tokens to capture longer-range dependencies without replacing the conv branch.

Also implement `trajectory_consistency_loss_v33(refined, raw, conf=None, smoothness_type="huber", delta=0.01)`:

- **Smoothness term:** Robust Huber (or L2) on the second-order finite difference of `refined`, masked by `conf`.
- **Drift term:** L2 distance between `refined` and `raw`, so the refiner cannot stray far from the original prediction.
- **Returns:** `{"smooth": ..., "drift": ..., "total": ...}` for logging.

### Changes to `motionflow_mv/fusion/omniview_fusion_v5.py`

Add the following constructor kwargs (mirroring the existing v32 block at lines ~199–202 and ~267–277):

- `use_trajectory_consistency_v33: bool = False`
- `v33_smooth_weight: float = 1e-3`
- `v33_drift_weight: float = 1e-2`
- `v33_robust_delta: float = 0.01`
- `v33_multi_scale: bool = True`
- `v33_attention_branch: bool = False`
- `v33_confidence_aware: bool = True`

In `forward`, after the residual/diffusion head and **before** the kinematic refiner (around the existing v32 usage at lines ~1105–1125):

1. Compute per-joint confidence from the input observations and view mask:
   ```python
   conf = (x[..., 2] * view_mask).sum(dim=2) / (view_mask.sum(dim=2) + 1e-8)  # (B, T, J)
   ```
2. Call `refined = self.trajectory_consistency_refiner_v33(pred_3d, conf=conf)`.
3. Compute `loss_dict = trajectory_consistency_loss_v33(refined, pred_3d, conf=conf)`.
4. Add `v33_smooth_weight * smooth + v33_drift_weight * drift` to `epi_loss`.
5. Replace `pred_3d` with `refined` for downstream heads.

The v32 and v33 flags should be mutually exclusive (raise `ValueError` if both are enabled).

### Changes to `experiments/train_omniview_fusion_v5_webbridge_multi.py`

Add argparse flags after the existing v32 block (~line 1437):

```python
parser.add_argument("--use_trajectory_consistency_v33", action="store_true", help="Use v33 confidence-aware multi-scale trajectory-consistency refiner")
parser.add_argument("--v33_smooth_weight", type=float, default=1e-3, help="Weight for v33 smoothness loss")
parser.add_argument("--v33_drift_weight", type=float, default=1e-2, help="Weight for v33 drift loss")
parser.add_argument("--v33_robust_delta", type=float, default=0.01, help="Huber delta for v33 robust smoothness")
parser.add_argument("--v33_multi_scale", action="store_true", default=True, help="Use multi-scale temporal convolutions in v33")
parser.add_argument("--no_v33_multi_scale", dest="v33_multi_scale", action="store_false")
parser.add_argument("--v33_attention_branch", action="store_true", default=False, help="Add lightweight temporal self-attention branch in v33")
parser.add_argument("--v33_confidence_aware", action="store_true", default=True, help="Use per-joint confidence masking in v33")
parser.add_argument("--no_v33_confidence_aware", dest="v33_confidence_aware", action="store_false")
```

Pass the new kwargs through `build_model_from_args` into `OmniMultiViewFusionV5`.

### Data / preprocessing requirements

No new dataset or loader is needed. Re-use the existing mixed-dataset manifest:

- `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`

The required per-view-joint confidences are already available as `x[..., 2]` and the view mask is already computed in `compute_loss` and passed to the model. The only preprocessing constraint is `clip_len >= 9` so that `T > 2` and the multi-scale kernels have enough temporal context.

## Training command / ablation flags

### Local RTX 4090 smoke test

```bash
python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
    --num_workers 0 --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --clip_len 9 --epochs 2 --batch_size 4 --train_samples 50 --val_stride 1 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 2 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 8 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --use_trajectory_consistency_v33 --v33_smooth_weight 1e-3 --v33_drift_weight 1e-2 \
    --v33_multi_scale --v33_confidence_aware --v33_robust_delta 0.01 \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 \
    --v29_physical_loss_warmup_epochs 2 \
    --output outputs/omniview_fusion_v33_trajectory_consistency_smoke_local_4090.pth
```

### Full A800 run

Replace the v32 flag in `scripts/launch_v32_a800_queue.py` with the v33 variant:

```bash
--use_trajectory_consistency_v33 --v33_smooth_weight 1e-3 --v33_drift_weight 1e-2 \
--v33_multi_scale --v33_confidence_aware --v33_robust_delta 0.01
```

### Suggested ablations

| Run | Extra flags | What it tests |
|-----|-------------|---------------|
| v33 baseline | `--use_trajectory_consistency_v33 --v33_smooth_weight 1e-3 --v33_drift_weight 1e-2 --v33_multi_scale --v33_confidence_aware` | Full v33 refiner |
| v33 no-confidence | add `--no_v33_confidence_aware` | Value of confidence-aware masking |
| v33 no-multi-scale | add `--no_v33_multi_scale` | Value of multi-scale kernels |
| v33 + attention | add `--v33_attention_branch` | Long-range temporal attention branch |
| v33 vs v32 | swap to `--use_trajectory_consistency_v32` | Direct comparison with v32 TCR |

Also sweep `v33_smooth_weight ∈ {1e-4, 1e-3, 1e-2}` at smoke scale to pick a stable value.

## Expected metrics and baseline to beat

- **Primary metric:** `val_MPJPE` on the mixed H36M + MPI-INF-3DHP validation set.
  - Baseline: v32 trajectory-consistency refiner (`--use_trajectory_consistency_v32`).
  - Target: match or improve v32 `val_MPJPE` (no regression) while simultaneously improving temporal smoothness.
- **Temporal smoothness metrics (computed on validation clips):**
  - Mean per-joint acceleration magnitude: `mean_j || Δ² pred_j ||_2`.
  - Acceleration error vs. ground truth: `mean_j || Δ² (pred_j − gt_j) ||_2`.
  - Target: `>= 10%` reduction in acceleration error vs. v32 while preserving MPJPE.
- **Robustness metrics:**
  - Variable-view MPJPE for `k = 2, 4, 8, 14` active views.
  - Outlier-view MPJPE with `--outlier_view_prob 0.3`.
  - Target: clean MPJPE preserved; `>= 2 mm` improvement over v32 at 30% view dropout/outlier rate.

## Risks / unknowns

1. **Over-smoothing fast motion.** Per-joint gates and robust Huber loss mitigate this, but athletic or very fast actions may still lose detail.
2. **Noisy confidence estimates.** If per-joint confidence from `x[..., 2]` is unreliable, confidence-aware masking can suppress correct joints or retain bad ones.
3. **Interaction with physical losses.** Strong temporal smoothing may mask the v28/v29 floor/bone losses or conflict with `v29_bone_temporal_weight`; need to ablate jointly.
4. **Attention-branch cost.** The optional transformer branch raises memory; if it OOMs at `clip_len=9`, it may need gradient checkpointing or removal.
5. **Mutual exclusivity with v32.** Enabling both v32 and v33 would double-smooth the output; the proposal explicitly forbids this, but the guard must be added to avoid silent regressions.
