# v33 Design Proposal: Hierarchical Multi-Scale Cross-View Spatial Pyramid

**Slug:** `hierarchical_multiscale`  
**Scope:** MotionFlow-MultiView v33 (next-iteration prototype)  
**Target downstream:** ICRA/CVPR 2027 multi-view pose pipeline  

## 1. Problem Statement and Motivation

Current v31/v32 cross-view fusion already reasons at multiple semantic scales—joint, part, and body—via `HierarchicalViewEncoderV30/V31`. However, this hierarchy is *single-resolution*: every scale operates on the full joint count `J`, and the part/body scales are formed by averaging joints rather than by an explicit spatial pyramid. At the same time, the older `SpatialFeaturePyramid` builds multi-resolution joint features but does not perform cross-view attention at each resolution, and the adaptive multi-scale fusion module (`AdaptiveHierarchicalMultiscaleFusion`) downsamples time and joints together without a skeleton-aware grouping.

**Goal for v33:** build a **hierarchical multi-scale cross-view spatial pyramid** that
1. decomposes the joint dimension into `J`, `J/2`, `J/4` (and optionally `J/8`) scales,
2. applies the stable v31 geometry-biased cross-view attention *at each scale* with scale-appropriate skeleton part groups,
3. fuses the multi-scale representations with per-token adaptive weights,
4. remains identity-at-init and compatible with variable-view training (`view_mask`).

The expected gain is better handling of fine-grained joints (wrists/ankles) and coarse limb/torso structure, while preserving robustness when views are dropped or corrupted.

## 2. Proposed Architecture Changes

### 2.1 New module: `HierarchicalMultiscaleCrossViewSpatialPyramidV33`

Location: `motionflow_mv/fusion/hierarchical_multiscale_spatial_pyramid_v33.py`

```python
class HierarchicalMultiscaleCrossViewSpatialPyramidV33(nn.Module):
    """Hierarchical multi-scale cross-view spatial pyramid.

    Args:
        d: token dimension.
        n_heads: attention heads for the per-scale cross-view blocks.
        n_views: maximum number of padded views (e.g. 14).
        scales: joint downsampling factors, default (1, 2, 4).
        n_part_layers: number of layers in the part-scale block.
        dropout: dropout inside cross-view attention.
        stochastic_depth_prob: stochastic depth probability.
        use_geometry_bias: whether to inject v31 epipolar/ray biases.
        use_adaptive_scale_fusion: whether to use per-token scale attention
            (True) or a fixed softmax weighting (False).
    """
```

Key components:

| Component | Role |
|-----------|------|
| `scale_pyramid` | For each scale `s`, downsample the joint dimension with adaptive average pooling (or strided 1-D conv) and upsample back to `J`. |
| `hierarchical_view_blocks` | One `HierarchicalViewEncoderV31`-style block per scale. Coarser scales use coarser part groups (e.g. limb/torso only). |
| `geometry_bias` | Reuses the v31 epipolar-distance + ray-intersection bias; computed once per scale from the same cameras/2D points. |
| `scale_attention` | Per-token attention over scales (`query = full-res token`, keys = per-scale outputs). |
| `residual_gate` | Sigmoid-initialised near zero so the block is identity at init. |

### 2.2 Model integration in `OmniMultiViewFusionV5`

In `motionflow_mv/fusion/omniview_fusion_v5.py`:

1. Add constructor flags:
   ```python
   use_hierarchical_multiscale_spatial_pyramid_v33: bool = False,
   v33_hmsp_scales: Sequence[int] = (1, 2, 4),
   v33_hmsp_n_heads: int = 4,
   v33_hmsp_n_part_layers: int = 1,
   v33_hmsp_dropout: float = 0.1,
   v33_hmsp_stochastic_depth_prob: float = 0.0,
   v33_hmsp_use_geometry_bias: bool = True,
   v33_hmsp_use_adaptive_scale_fusion: bool = True,
   ```

2. If the flag is enabled, instantiate the new module after the existing `hierarchical_multiview_v31` block (or as a replacement for it) and before the spatio-temporal transformer stage:
   ```python
   if self.use_hierarchical_multiscale_spatial_pyramid_v33:
       self.hierarchical_multiscale_spatial_pyramid_v33 = (
           HierarchicalMultiscaleCrossViewSpatialPyramidV33(
               d=self.d,
               n_heads=v33_hmsp_n_heads,
               n_views=n_views,
               scales=v33_hmsp_scales,
               n_part_layers=v33_hmsp_n_part_layers,
               dropout=v33_hmsp_dropout,
               stochastic_depth_prob=v33_hmsp_stochastic_depth_prob,
               use_geometry_bias=v33_hmsp_use_geometry_bias,
               use_adaptive_scale_fusion=v33_hmsp_use_adaptive_scale_fusion,
           )
       )
   ```

3. In `forward()`, pass the view mask and camera tensors:
   ```python
   if self.use_hierarchical_multiscale_spatial_pyramid_v33:
       feat = feat + self.hierarchical_multiscale_spatial_pyramid_v33(
           feat,
           view_mask=view_mask_flat.view(B, T, V),
           points_2d=points_2d.view(B, T, V, J, 2),
           K=K_corrected.view(B, T, V, 3, 3),
           R=R.view(B, T, V, 3, 3),
           t=t.view(B, T, V, 3),
       )
   ```

### 2.3 Skeleton part groups per scale

Reuse the existing `H36M_17_PART_GROUPS` / `MPI_28_PART_GROUPS` from `hierarchical_multiview_v30.py` at full resolution. At coarser scales, collapse adjacent parts:

- Scale 1 (`J`): head / torso / arms / legs.
- Scale 2 (`J/2`): upper body / lower body.
- Scale 4 (`J/4`): single body token.

This is analogous to v30/v31 but applied on downsampled joint tensors, allowing the model to mix coarse spatial context before the fine ST transformer.

## 3. Training Command / Ablation Flags

### Recommended smoke test (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
  --use_domain_embedding \
  --use_multiscale_fusion true --use_camera_conditioning --use_epipolar_bias \
  --use_context_visibility --use_skeleton_residual --use_rotation_correction \
  --use_entropy_regularization --attention_entropy_weight 0.01 \
  --use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention \
  --use_hierarchical_multiscale_spatial_pyramid_v33 \
  --v33_hmsp_scales 1 2 4 \
  --v33_hmsp_n_part_layers 1 \
  --v33_hmsp_dropout 0.1 \
  --v33_hmsp_use_geometry_bias \
  --v33_hmsp_use_adaptive_scale_fusion \
  --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
  --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
  --d 64 --residual_hidden 128 --n_st_layers 2 --graph_num_layers 1 \
  --epochs 5 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --output outputs/omniview_fusion_v33_hierarchical_multiscale_smoke.pth
```

### Ablation matrix

| Run | Flags | Purpose |
|-----|-------|---------|
| `v32_baseline` | `--use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention` | Current v31/v32 best practice. |
| `v33_hmsp_full` | baseline + `--use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4` | Full pyramid at three scales. |
| `v33_hmsp_no_adaptive` | full + `--v33_hmsp_use_adaptive_scale_fusion` disabled | Test fixed softmax scale weights. |
| `v33_hmsp_no_geometry` | full + `--v33_hmsp_use_geometry_bias` disabled | Test geometry bias contribution. |
| `v33_hmsp_scales_1_2` | full but `--v33_hmsp_scales 1 2` | Reduce pyramid depth / compute. |

## 4. Expected Metrics and Baseline to Beat

### Primary metrics

| Metric | How computed | Target |
|--------|--------------|--------|
| `val_MPJPE` | `eval_metric()` in `train_omniview_fusion_v5_webbridge_multi.py` (mm) | **Beat v32 baseline** (recorded v32 runs ~20–40 mm on smoke; full A800 target < 28 mm). |
| `val_MPJPE_h36m` | Per-domain validation subset (`dataset_id == 0`) | Improve or match v31 baseline on 4-view H36M. |
| `val_MPJPE_mpi` | Per-domain validation subset (`dataset_id == 1`) | Improve or match v31 baseline on 14-view MPI. |

### Robustness metrics

Use the existing variable-view and outlier-view paths:

```python
for k in [2, 4, 8, 14]:
    mask = sample_k_views(k)
    mpjpe = evaluate(view_mask=mask)
```

| Metric | Target |
|--------|--------|
| Relative degradation at `k=2` vs `k=14` | ≤ 10% increase in MPJPE. |
| Outlier-view MPJPE | ≤ 5% worse than clean when one view is corrupted (reuse `--outlier_view_prob` augment). |
| Single-clip latency | ≤ 1.3× v31 baseline on RTX 4090 at `d=64`. |

### Baseline to beat

- **v31 hierarchical encoder** (`--use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention`).
- **v32 combined** run in `scripts/launch_v32_a800_queue.py` (`v32_combined`), which adds domain-aware view curriculum + trajectory consistency.

## 5. Risks / Unknowns

1. **Pooling blur.** Adaptive pooling over joints may remove fine spatial detail at coarse scales.
   - *Mitigation:* use strided 1-D convolutions instead of pooling; keep the finest scale as identity branch.
2. **Geometry bias cost.** Computing v31 epipolar/ray biases at multiple scales increases memory and runtime.
   - *Mitigation:* share the geometry bias computed at full resolution and downsample it; only compute at coarse scales if profiling allows.
3. **Variable-view padding.** The pyramid assumes a fixed `J`; the mixed loader pads H36M to 14 views but keeps `J=17` or `28`, so joint downscaling is safe. The `view_mask` must be applied inside every per-scale block.
4. **Scale-fusion instability.** Per-token scale attention can overfit on small smoke sets.
   - *Mitigation:* start with fixed softmax weights (`use_adaptive_scale_fusion=False`) and enable adaptive weights only after the smoke succeeds.
5. **No improvement over v31.** The pyramid adds parameters (~`d²` per scale) without guaranteed accuracy gains.
   - *Mitigation:* the module is identity-at-init and gated; if it fails, drop the flag and training reverts to the v31 path.

## 6. Success Criteria (Go/No-Go)

- Smoke test completes without NaNs and `val_MPJPE` is within 5% of the v31 baseline on the same smoke config.
- Full A800 run matches or beats v32 `val_MPJPE` (< 28 mm).
- Variable-view robustness curve is flatter than v31 by at least 10% AUC.

## 7. Files Touched (if/when implemented)

- `motionflow_mv/fusion/hierarchical_multiscale_spatial_pyramid_v33.py` (new)
- `motionflow_mv/fusion/omniview_fusion_v5.py` (add flag + wiring)
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` (add CLI flags)
- `docs/swarm_iter_next/v33_hierarchical_multiscale.md` (this proposal)

---

*This proposal does not modify any existing source files; it only describes the intended v33 integration path.*
