# v33 Multi-Task Learning with 2D/3D Keypoint Supervision

**Direction slug:** `multitask_keypoint_supervision`  
**Target model:** `OmniMultiViewFusionV5` (`motionflow_mv/fusion/omniview_fusion_v5.py`)  
**Training entry point:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`  

## 1. Problem statement and motivation

`OmniMultiViewFusionV5` currently optimises a single 3-D pose objective: the final triangulated/refined joints are supervised against ground-truth 3-D poses.  All 2-D information enters only implicitly through reprojection losses and triangulation.  This leaves useful supervisory signals on the table:

- **Per-view 2-D keypoints** are available at every frame, but the network never has an explicit 2-D keypoint decoding task.
- **Intermediate 3-D representations** (after graph-joint attention, after the ST transformer) are only weakly constrained; adding 3-D keypoint supervision at intermediate stages can stabilise gradients and improve feature quality.
- **Per-joint confidence/visibility** is predicted, but not directly supervised against the input confidence channel in a multi-task fashion.

A dedicated multi-task 2D/3D keypoint supervision branch should give the encoder richer, more stable gradients, especially under heavy occlusion and variable-view training, and should transfer well to variable-view inference without changing the inference pipeline.

## 2. Proposed architecture changes

### 2.1 New module: `KeypointMultiTaskHeadV33`

Create `motionflow_mv/fusion/multitask_keypoint_supervision_v33.py` with a single module:

```python
class KeypointMultiTaskHeadV33(nn.Module):
    def __init__(self, d: int, j: int, n_views: int, hidden: int = 128):
        ...

    def forward(self, feat, points_2d_input, confidences):
        """
        Args:
            feat: (B*T, V, J, d) fused feature tokens.
            points_2d_input: (B*T, V, J, 2) raw 2-D observations.
            confidences: (B*T, V, J) visibility weights.
        Returns:
            pred_2d_residual: (B*T, V, J, 2) residual added to input 2-D points.
            pred_2d_conf: (B*T, V, J) predicted per-view 2-D keypoint confidence.
            pred_3d_intermediate: (B*T, J, 3) 3-D pose from pooled features.
        """
```

The head is deliberately lightweight: two MLPs per joint/view for 2-D residuals, a small view-pooling + MLP for intermediate 3-D, and a confidence logit head.

### 2.2 Integration into `OmniMultiViewFusionV5`

Add the following constructor arguments to `OmniMultiViewFusionV5.__init__`:

- `use_multitask_keypoint_supervision_v33: bool = False`
- `v33_keypoint_2d_loss_weight: float = 0.1`
- `v33_keypoint_3d_loss_weight: float = 0.1`
- `v33_confidence_loss_weight: float = 0.05`
- `v33_intermediate_3d_supervision: bool = True`

When enabled, the model instantiates `KeypointMultiTaskHeadV33` and:

1. After graph-joint attention + camera conditioning (`motionflow_mv/fusion/omniview_fusion_v5.py` lines ~780–784), feeds the feature `feat` into the head to produce:
   - `pred_2d_residual` (added to input `points_2d` to obtain corrected 2-D keypoints),
   - `pred_2d_conf` (supervised against input confidences),
   - `pred_3d_intermediate` (supervised against ground-truth 3-D pose).
2. The corrected 2-D keypoints are optionally used as an additional residual input to the triangulation module, but the main pipeline remains unchanged to avoid destabilising inference.
3. The head’s losses are returned in the model output tuple as the 6th element:
   ```python
   out = (pred_3d, weights, visibility, L, epi_loss, v33_keypoint_loss)
   ```

### 2.3 Losses in `build_compute_loss`

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, when `args.use_multitask_keypoint_supervision_v33` is true and `out[5]` is present:

```python
v33_loss = out[5]
loss = loss + v33_loss
metrics["v33_keypoint_loss"] = v33_loss.item()
```

Internally, `KeypointMultiTaskHeadV33.forward` computes:

```
L_v33 = λ_2d * MSE(pred_2d_corrected, gt_2d) * visible_mask
      + λ_3d * MSE(pred_3d_intermediate, gt_3d)
      + λ_conf * BCE(pred_2d_conf, visible_mask)
```

The 2-D loss is computed in pixels and normalised by focal length to put it on a comparable scale to the 3-D loss.  The intermediate 3-D loss is detached from the final prediction so that the main 3-D branch remains the primary target.

## 3. Training command / ablation flags

### 3.1 Recommended smoke command

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_multitask_keypoint_supervision_v33 \
    --v33_keypoint_2d_loss_weight 0.1 \
    --v33_keypoint_3d_loss_weight 0.1 \
    --v33_confidence_loss_weight 0.05
```

### 3.2 Recommended full WebBridge + H36M + MPI command

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 \
    --use_multitask_keypoint_supervision_v33 \
    --v33_keypoint_2d_loss_weight 0.1 \
    --v33_keypoint_3d_loss_weight 0.1 \
    --v33_confidence_loss_weight 0.05 \
    --v33_intermediate_3d_supervision true \
    --d 64 --residual_hidden 128 --n_st_layers 2 --graph_num_layers 1 --n_joint_layers 1 \
    --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0
```

### 3.3 Ablation flags

| Flag | Purpose |
|------|---------|
| `--use_multitask_keypoint_supervision_v33` | Enable the multi-task keypoint head. |
| `--v33_keypoint_2d_loss_weight W` | Weight for 2-D keypoint correction loss. |
| `--v33_keypoint_3d_loss_weight W` | Weight for intermediate 3-D keypoint supervision. |
| `--v33_confidence_loss_weight W` | Weight for per-view 2-D confidence BCE loss. |
| `--v33_intermediate_3d_supervision {true,false}` | Toggle intermediate 3-D supervision while keeping 2-D task. |
| `--v33_keypoint_head_hidden D` | Hidden dim of the 2-D/3-D keypoint MLPs (default 128). |

## 4. Expected metrics and baseline to beat

The primary comparison is against the current v32 best local/A800 baseline (`v32_combined` / `v30a`), which reports val_MPJPE in the 20–45 mm range on mixed WebBridge/H36M/MPI smoke/full runs.

| Metric | Target |
|--------|--------|
| `val_MPJPE` on mixed H36M/MPI val | Beat the v32 baseline by ≥1 mm on smoke; ≥3 mm on full. |
| `val_MPJPE` under variable-view (2–14 views) | Lower degradation when trained with `--use_variable_view_training`. |
| 2-D keypoint PCK @0.05 (pixels) | Track as a diagnostic; no hard threshold, but should improve monotonically. |
| Occlusion robustness | Lower MPJPE on clips with ≥30% occluded joints after `--occlusion_augment_prob 0.3`. |

Baseline to beat for smoke: any existing v32 smoke run that completes with <30 mm val_MPJPE.  Baseline to beat for full: the `v32_combined` / `v30a` full run.

## 5. Risks / unknowns

1. **Loss balance.** The 2-D keypoint loss operates in pixel/focal-normalised units while the 3-D loss is in metres.  Getting `λ_2d` and `λ_3d` wrong can either drown the main 3-D objective or provide no useful gradient.  A simple warmup/ramp for the new losses is recommended.
2. **Feature hijacking.** The encoder may learn to satisfy the auxiliary 2-D/3-D tasks at the expense of the final triangulation.  Detaching the intermediate 3-D loss from the final output and using a low weight mitigates this.
3. **Variable-view compatibility.** The head must ignore padded/masked views.  The existing `view_mask` mechanism handles this, but the 2-D confidence target needs to be zeroed for masked views.
4. **Computational cost.** The new head is lightweight, but adds per-view MLPs.  Smoke-test first on RTX 4090 to confirm no OOM or significant slowdown.
5. **Overlap with existing reprojection loss.** The 2-D keypoint task is related to the existing `--reproj_loss_weight` and `--aleatoric_reproj_loss_weight`.  Ablate whether the multi-task head adds value beyond simply increasing the reprojection weight.

## 6. Implementation checklist (for the v33 developer)

- [ ] Create `motionflow_mv/fusion/multitask_keypoint_supervision_v33.py`.
- [ ] Add v33 flags to `OmniMultiViewFusionV5.__init__` and instantiate the head.
- [ ] Wire the head into the forward pass at the post-graph-joint/camera-conditioning feature stage.
- [ ] Return the auxiliary loss as `out[5]`.
- [ ] Add CLI flags and loss accumulation in `experiments/train_omniview_fusion_v5_webbridge_multi.py`.
- [ ] Smoke test with `--smoke` on CPU and a short RTX 4090 run.
- [ ] Ablate 2-D only, 3-D only, and combined settings.
