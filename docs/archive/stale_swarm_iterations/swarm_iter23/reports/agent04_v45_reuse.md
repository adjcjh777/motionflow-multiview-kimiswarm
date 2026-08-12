# v45-AGF Reliability Weights Reuse for v46 Sparse-View Generalization

**Agent:** Agent-04 (ANALYZE)  
**Date:** 2026-08-09  
**Issue:** #160  
**Scope:** Review `AdaptiveGeometryFusionV45` and identify how its reliability weights can be reused by the v46 Sparse-View Generalization (SVG) module.  

## 1. What v45-AGF currently does

`motionflow_mv/fusion/adaptive_geometry_fusion_v45.py::AdaptiveGeometryFusionV45` is a small MLP head that predicts per-view (or per-joint / per-view-joint) reliability weights from reprojection residuals. Key facts from the code:

- **Inputs:** `points_2d (B,T,V,J,2)`, an initial 3D estimate `pred_3d (B,T,J,3)`, and camera parameters `K, R, t`. Optional `view_mask (B,T,V)`.
- **Output:** `weights (B,T,V,J)` of positive reliability weights, zero for masked views.
- **Mechanism:** It computes per-(view,joint) reprojection residuals, pools statistics according to `weight_type`, feeds them through a tiny MLP, and maps the result with `2 * sigmoid(x)` to a positive weight.
- **Initialization:** The final layer is zero-initialized so the module starts as a near-identity (`weight ≈ 1.0`).
- **Integration:** In `MultiViewGeometryFusionV25.forward` (lines 517–528), when `use_v45_adaptive_geometry_fusion` is true, the v45 weights are multiplied into the triangulation confidence and DLT is re-run:

```python
v45_weights = self.adaptive_geometry_fusion_v45(..., view_mask=view_mask)
confidence = confidence * v45_weights
tri_weights = confidence if view_mask is None else confidence * view_mask[:, :, :, None]
pred_3d_init = triangulate_initial(pts, K, R, t, weights=tri_weights)
```

This already makes the downstream triangulation aware of view validity.

## 2. Why v45 is already a v46 "reliability head"

The v46 proposal (`docs/proposals/v46_sparse_view_generalization.md`) describes:

> *Per-view reliability head predicts reliability r_v ∈ (0,1) for each available view, masked to 0 for dropped/missing views.*

`AdaptiveGeometryFusionV45` satisfies this almost exactly:

| v46 requirement | v45 capability |
|-----------------|----------------|
| Per-view reliability | `weight_type="per_view"` (or `"per_view_joint"`) |
| Masked views get zero weight | `if view_mask is not None: weights = weights * mask` |
| Positive bounded weights | `2 * sigmoid(...)` + clamp |
| Compatible with weighted DLT | Output shape `(B,T,V,J)` matches `confidence` and `triangulate_initial` |
| Identity at start | Zero-initialized final layer => weight ≈ 1.0 |

Therefore, the v46 SparseViewGeneralizationV46 module **does not need to train a new reliability MLP**; it can directly reuse v45 as the reliability head.

## 3. Recommended reuse patterns

### Pattern A: Use v45-AGF as-is inside v25 geometry fusion (minimal change)

- Keep `use_v45_adaptive_geometry_fusion=True` in the model / trainer.
- When v46 randomly drops views during training, pass the updated `view_mask` into the existing v45 call.
- v45 will automatically down-weight or zero-out the dropped views and re-triangulate with the surviving ones.

**Pros:** No new code path; v45 already returns `(B,T,V,J)` weights that feed `triangulate_initial`.  
**Cons:** v45 currently needs an initial 3D estimate `pred_3d_init` to compute residuals. With only two views, the initial DLT can be noisy, so the learned weights may be less reliable. A fallback to uniform weights when `available_views < min_views` should be retained.

### Pattern B: Wrap v45 inside `SparseViewGeneralizationV46`

`motionflow_mv/fusion/sparse_view_generalization_v46.py` can instantiate `AdaptiveGeometryFusionV45` and call it as its *per-view reliability head* rather than implementing a new one:

```python
from motionflow_mv.fusion.adaptive_geometry_fusion_v45 import AdaptiveGeometryFusionV45

class SparseViewGeneralizationV46(nn.Module):
    def __init__(self, n_views: int, hidden: int = 64, ...):
        super().__init__()
        self.reliability_head = AdaptiveGeometryFusionV45(
            n_views=n_views,
            weight_type="per_view_joint",  # most expressive, but per_view is lighter
            hidden=hidden,
            n_layers=1,
        )

    def forward(self, points_2d, pred_3d, K, R, t, view_mask):
        # v45 returns (B,T,V,J); v46 proposal expects (B,T,V,J) or (B,T,V)
        weights = self.reliability_head(points_2d, pred_3d, K, R, t, view_mask=view_mask)
        return weights  # downstream v25 DLT uses these as r_v
```

This keeps the v46 module small and avoids duplicating the residual-computation logic.

### Pattern C: Use v45 weights as a prior for the view-agnostic set aggregator

The proposal also calls for a *view-agnostic set aggregator* (ISAB-based, already implemented in `VariableViewSetAggregator`). v45 weights can be used to gate the aggregator tokens before/after the ISAB layers:

```python
# Inside SparseViewGeneralizationV46
reliability = self.v45_reliability(...)        # (B,T,V,J)
tokens = self.ray_tokenizer(...)                # (B,T,V,J,d)
gated_tokens = tokens * reliability[..., None]  # down-weight noisy views
aggregated = self.isab(gated_tokens, view_mask) # (B,T,V,J,d)
```

This matches the v46 architecture diagram where the reliability head feeds both the ISAB and the weighted triangulation.

## 4. Integration points in the existing pipeline

- `motionflow_mv/fusion/omniview_fusion_v5.py` already passes `use_v45_adaptive_geometry_fusion` and related flags into `MultiViewGeometryFusionV25`.
- `MultiViewGeometryFusionV25.forward` already applies `view_mask` to the v45 weights and to the triangulation.
- The v46 view-dropout augmentation (`motionflow_mv/data/view_dropout_augmentation_v46.py`) only needs to output a boolean `view_mask`; the existing v45 path will handle the weighting.
- `experiments/eval_variable_views.py` can reuse the same mask to report `MPJPE@k` by clamping the mask to `k` views.

## 5. Caveats and open questions

1. **Minimum number of views.** v45's residual-based weights require a 3D estimate. With `< 2` views, triangulation is ill-posed. The v46 helper must enforce `min_views >= 2` and fall back to uniform weights when too few views remain.
2. **Interaction with `OutlierViewDetector`.** v25 geometry fusion can run both v45 and the outlier detector. The order today is v45 first, then outlier re-weighting. v46 should preserve this order (or document a new one) to avoid double suppression.
3. **Per-view vs. per-view-joint.** The proposal says *per-view reliability r_v*, but v45 returns per-joint weights when `weight_type="per_joint"`. For v46, `weight_type="per_view"` or `"per_view_joint"` is the natural choice.
4. **Test coverage.** Existing tests in `tests/test_adaptive_geometry_fusion_v45.py` already verify shape, positivity, masking, and DLT integration. v46 should extend these with variable-view masks and `MPJPE@k` checks rather than re-implementing the same module tests.

## 6. Conclusion

The v45-AGF reliability weights can be reused directly as the v46 *per-view reliability head*. The cleanest path is:

1. Keep `AdaptiveGeometryFusionV45` unchanged.
2. Have `SparseViewGeneralizationV46` call it (Pattern B or C) rather than adding a new MLP.
3. Drive the module with the v46 view-dropout `view_mask`, so dropped views receive zero weight.
4. Feed the resulting `(B,T,V,J)` weights back into the existing weighted DLT path in `MultiViewGeometryFusionV25`.

This avoids duplicating residual-computation and triangulation logic and keeps the v46 change minimal and consistent with the v45/v25 design.
