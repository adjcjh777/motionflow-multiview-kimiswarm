# v46-SVG Integration Analysis for v47 Temporal Aggregation Head

**Agent:** Agent-02 (ANALYZE)  
**Date:** 2026-08-09  
**Branch:** `v47-temporal`  
**Tracking issue:** #162  

## Goal

Identify the exact place in the v46-SVG pipeline where the `TemporalAggregationV47` head should be wired into `OmniMultiViewFusionV5`, and what data it needs from the v46 modules.

## Files reviewed

- `motionflow_mv/fusion/sparse_view_generalization_v46.py`
- `motionflow_mv/data/view_dropout_augmentation_v46.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` (view-mask / dropout path only)

## How v46-SVG currently works

### 1. `SparseViewGeneralizationV46` in `omniview_fusion_v5.py`

**Instantiation** (`omniview_fusion_v5.py`, lines 757–773):

```python
self.use_v46_sparse_view_generalization = use_v46_sparse_view_generalization
self.v46_svg_view_dropout_prob = v46_svg_view_dropout_prob
self.v46_svg_min_views = max(2, v46_svg_min_views)
self.v46_svg_hidden = v46_svg_hidden
self.v46_svg_use_curriculum = v46_svg_use_curriculum
if self.use_v46_sparse_view_generalization:
    from motionflow_mv.fusion.sparse_view_generalization_v46 import (
        SparseViewGeneralizationV46,
    )

    self.sparse_view_generalization_v46 = SparseViewGeneralizationV46(
        in_channels=self.d,
        n_views=n_views,
        hidden=v46_svg_hidden,
    )
else:
    self.sparse_view_generalization_v46 = None
```

**Forward call** (`omniview_fusion_v5.py`, lines 1083–1092):

```python
# Optional v46 sparse-view generalization reliability head.
v46_reliability = None
if (
    self.use_v46_sparse_view_generalization
    and self.sparse_view_generalization_v46 is not None
):
    v46_reliability = self.sparse_view_generalization_v46(
        feat,
        view_mask=view_mask_flat.view(B, T, V).bool(),
    )  # (B, T, V, J)
```

**Application to triangulation weights** (`omniview_fusion_v5.py`, lines 1326–1330):

```python
# Optional v46 sparse-view generalization: down-weight dropped/missing views.
if v46_reliability is not None:
    # v46_reliability is (B, T, V, J); weights is (B*T, V, J).
    weights = weights * v46_reliability.view(B * T, V, J)
    weights = weights.clamp(min=1e-4, max=1e4)
```

So v46 produces **per-view, per-joint reliability weights** that multiply the DLT weights. It does **not** produce a final 3-D pose; that is still computed by the normal triangulation path.

### 2. `ViewDropoutAugmentationV46` in the trainer

The trainer applies view dropout in `augment_clip` (`train_omniview_fusion_v5_webbridge_multi.py`, lines 989–1010):

```python
x, view_mask = augment_clip(
    x.clone(),
    noise_std=args.noise_std if not args.smoke else 0.0,
    dropout_rate=args.confidence_dropout,
    view_dropout_rate=v46_dropout_prob,
    min_views=v46_min_views,
    variable_view_subset=args.variable_view_subset,
)
```

- `view_mask` has shape `(B, V)` or `(B, T, V)` after internal expansion.
- The confidence channel of `x` is zeroed for dropped views, so the model sees real sparse-view inputs.

## Exact integration point for v47

The v47 head is a **post-triangulation temporal smoother** that operates on the final per-frame 3-D pose. The natural insertion point is immediately after the residual MLP produces the per-frame triangulated pose, and **before** the optional v32 trajectory-consistency refiner so that subsequent optional heads (v32, kinematic refiner, v19 temporal perceiver, v28 physical alignment) can consume the temporally refined pose.

### Location

In `motionflow_mv/fusion/omniview_fusion_v5.py`, in `OmniMultiViewFusionV5.forward`, between:

- **After** line 1558: `pred_3d = pred_3d_gn + delta`
- **Before** line 1561: the v32 `use_trajectory_consistency_v32` block

At this point in the forward pass:

- `pred_3d` has shape `(B * T, J, 3)`.
- `view_mask_flat` has shape `(B * T, V)` and is normalized.
- `B`, `T`, `V`, `J` are already defined.

### Suggested wiring (reference patch for IMPLEMENT agents)

```python
# -------------------------------------------------------------------------
# v47 temporal aggregation head (post-triangulation smoother)
# -------------------------------------------------------------------------
self.use_v47_temporal_aggregation = use_v47_temporal_aggregation
self.v47_temporal_d_model = v47_temporal_d_model
self.v47_temporal_n_heads = v47_temporal_n_heads
self.v47_temporal_num_layers = v47_temporal_num_layers
self.v47_temporal_window = v47_temporal_window
self.v47_temporal_dropout = v47_temporal_dropout
self.v47_temporal_loss_weight = v47_temporal_loss_weight
self.v47_use_view_count_conditioning = v47_use_view_count_conditioning
if self.use_v47_temporal_aggregation:
    from motionflow_mv.fusion.temporal_aggregation_v47 import TemporalAggregationV47
    self.temporal_aggregation_v47 = TemporalAggregationV47(
        n_joints=self.j,
        d_model=v47_temporal_d_model,
        n_heads=v47_temporal_n_heads,
        num_layers=v47_temporal_num_layers,
        temporal_window=v47_temporal_window,
        dropout=v47_temporal_dropout,
        use_view_count_conditioning=v47_use_view_count_conditioning,
    )
else:
    self.temporal_aggregation_v47 = None
```

**Forward insertion** (`omniview_fusion_v5.py`, after line 1558 and before v32 block):

```python
# Optional v47 temporal aggregation (post-triangulation temporal smoother).
if (
    self.use_v47_temporal_aggregation
    and self.temporal_aggregation_v47 is not None
    and T > 1
):
    pred_3d_seq = pred_3d.view(B, T, self.j, 3)
    pred_3d_seq = self.temporal_aggregation_v47(
        poses_3d=pred_3d_seq,
        view_mask=view_mask_flat.view(B, T, V),
        clip_mask=None,  # or view_mask_flat.view(B, T, V).any(dim=-1)
    )
    pred_3d = pred_3d_seq.view(B * T, self.j, 3)
```

### Why this exact location?

1. **Per-frame triangulation is complete.** `pred_3d_gn + delta` is the final per-frame 3-D estimate produced by the v25/v45/v46 triangulation stack. Inserting v47 earlier would bypass the v46 reliability-weighted triangulation; inserting later would duplicate work already done by v32/v19/v28.
2. **v46 reliability already flowed into the pose.** The reliability weights from `SparseViewGeneralizationV46` were multiplied into `weights` before DLT, so the resulting `pred_3d` is the v46-aware pose. The v47 head therefore sees the best per-frame estimate available.
3. **Preserves optional downstream heads.** v32 trajectory consistency, v19 temporal perceiver, v28 physical alignment, and the kinematic refiner all consume `pred_3d`. Running v47 before them lets the whole downstream stack benefit from temporal smoothing.
4. **Easy identity-at-init.** `TemporalAggregationV47` is designed to be a no-op when its residual gate is zero. Inserting it here means `pred_3d` is unchanged at initialization, so existing v46 checkpoints load without regression.

### Inputs available at the insertion point

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `pred_3d` | `(B*T, J, 3)` | Per-frame triangulated pose after residual MLP |
| `view_mask_flat` | `(B*T, V)` | Binary view mask used by v46 |
| `B, T, V, J` | scalars | Batch, temporal, view, joint dimensions |
| `epi_loss` | scalar | Current loss accumulator; v47 smoothness loss can be added here |

To call the proposed `TemporalAggregationV47.forward`:

- `poses_3d` → `pred_3d.view(B, T, J, 3)`
- `view_mask` → `view_mask_flat.view(B, T, V)`
- `clip_mask` → `view_mask_flat.view(B, T, V).any(dim=-1)` (True = frame has at least one active view)

### Training-loop considerations

- A new flag `--use_v47_temporal_aggregation` should be added to the trainer alongside the v46 flags.
- The trainer already applies v46 view dropout, so the `view_mask` passed to the model already encodes the sparse-view configuration the v47 head should trust.
- v47 temporal smoothness loss:

```python
loss_temporal = v47_temporal_loss_weight * mean(|P_t - P_{t-1}|)
```

  should be computed inside `TemporalAggregationV47` (or in `forward`) and added to `epi_loss`.

- When `T == 1` (single-frame inference), the v47 head should be skipped because there is no temporal context.

### Potential conflict with v32

`use_trajectory_consistency_v32` is another temporal smoother. If both flags are enabled, the recommended order is:

1. v47 transformer-based temporal aggregation (post-triangulation, full-clip or local-window attention)
2. v32 trajectory-consistency refiner (optimization-based smoothness/drift penalty)

This keeps v47 as the primary temporal evidence aggregator and v32 as an optional regularizer. If v32 is deprecated for v47 runs, the v47 block can simply replace it.

## Tests run

Ran the v46 module smoke tests to confirm the current v46-SVG code is functional before v47 wiring:

```bash
source .venv/bin/activate
python -m motionflow_mv.fusion.sparse_view_generalization_v46
python motionflow_mv/data/view_dropout_augmentation_v46.py
```

Results:

```
SparseViewGeneralizationV46 CPU smoke test passed
view_dropout_augmentation_v46 smoke tests passed
```

## Deliverable

This report confirms that the `TemporalAggregationV47` head should be inserted in `motionflow_mv/fusion/omniview_fusion_v5.py` immediately after the residual MLP at line 1558 (`pred_3d = pred_3d_gn + delta`) and before the optional v32 trajectory-consistency block. It consumes the v46-aware per-frame pose `pred_3d` and the existing `view_mask_flat`, requires no changes to the v46 modules, and is gated by a new `use_v47_temporal_aggregation` flag.
