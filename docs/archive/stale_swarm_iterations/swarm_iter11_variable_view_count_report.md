# Iter11+ Roadmap: Variable View Count

## Executive Summary

MotionFlow-MultiView is currently trained and evaluated with a fixed number of cameras (`n_views=4`). I verified that the newest combined model (`RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`) already tolerates `V <= n_views` because it slices positional embeddings (`view_pos_embed[:V]`), but it crashes for `V > n_views`. Real deployments have drop-in/drop-out cameras, mobile rigs with 2–6 views, and datasets such as AIST++ (9 views) or MPI-INF-3DHP (up to 14 views). Making the architecture natively variable-view is a concrete, achievable Iter11+ goal that can improve both MPJPE on partial views and the ICRA/CVPR story.

This report proposes minimal engineering changes, training augmentations, and evaluation protocols. The target is a single model that (a) matches or beats the 11.17 mm MPJPE baseline when 4 views are present, (b) degrades gracefully down to 2 views, and (c) generalizes across datasets with different camera counts without retraining.

## Current Bottlenecks

1. **Fixed `view_pos_embed`**: `RayAttentionFusionModelTemporalCrossview` stores `nn.Parameter(torch.randn(n_views, d))`. Inputs with more views than `n_views` raise a shape error.
2. **Fixed-size fusion MLP**: The same base model has `nn.Linear(d * n_views, d)`. The newest combined model does not use it, but it is present in older cross-view variants and would break for variable `V`.
3. **No padding/attention mask**: Training collate functions assume every sample has the same `V`, so mixed batches with different view counts are impossible.
4. **Rigid dataset converters**: H36M, MPI, and AIST++ converters assume all views are present; there is no generic view-subsampling path.
5. **Unmasked triangulation**: DLT and Gauss-Newton heads assume all views are valid; missing views currently corrupt the result.

## Proposed Changes

1. **Replace `n_views` with `max_n_views` and slice embeddings at runtime.** This is already done for `time_pos_embed` but not for `view_pos_embed`.
2. **Replace the fixed-size view-concatenation `fusion_mlp` with per-token pooling** (mean/max over views) or an attention-based aggregator. The combined model already pools features before the residual head, so this is the natural design.
3. **Add `view_mask` support for mixed batches.** Use the mask in multi-head attention, positional embedding slicing, and triangulation so padded views are ignored.
4. **Add view-dropout augmentation and a view-count curriculum.** Randomly drop entire views during training to simulate a variable-view regime without collecting new data.
5. **Make triangulation robust to missing views.** Zero-weight padded/missing views before DLT and Gauss-Newton, and guard against `V < 2`.

## Code Sketch

```python
class RayAttentionFusionModelVariableView(nn.Module):
    def __init__(self, j=17, d=64, max_n_views=16, n_heads=4,
                 n_joint_layers=1, n_st_layers=2):
        super().__init__()
        self.j = j
        self.d = d
        self.max_n_views = max_n_views

        # capacity-sized embeddings, sliced by actual V
        self.view_pos_embed = nn.Parameter(torch.randn(max_n_views, d) * 0.02)
        self.time_pos_embed = nn.Parameter(torch.randn(256, d) * 0.02)

        # per-token aggregation instead of d*V concatenation
        self.view_pool = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.weight_head = nn.Linear(d, 1)

        self.residual_mlp = nn.Sequential(
            nn.Linear(d + 3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.st_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d, n_heads, d * 2,
                                       batch_first=True, norm_first=True)
            for _ in range(n_st_layers)
        ])

    def forward(self, x, view_mask=None, K=None, R=None, t=None):
        # x: (B, T, V, J, 3), V <= max_n_views
        B, T, V, J, _ = x.shape

        # slice positional embeddings by actual view count
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)

        feat = self._extract_frame_features(x, K, R, t)  # (B*T, V, J, d)
        feat = feat.view(B, T, V, J, self.d) + time_emb + view_emb
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)

        for layer in self.st_transformer:
            feat = layer(feat, src_key_padding_mask=view_mask)

        # pool per-view tokens and predict weights
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        weights = torch.sigmoid(self.weight_head(feat_pooled)).permute(0, 2, 1)
        if view_mask is not None:
            weights = weights * view_mask[..., None]

        pred = self._triangulate_and_refine(feat_pooled, weights, K, R, t)
        return pred
```

Key points: `view_pos_embed[:V]` removes the hard-coded view count, `src_key_padding_mask` handles mixed batches, and the triangulation head zeroes padded views.

## Recommended Experiments

1. **Variable-view training on MPI-INF-3DHP.** Set `max_n_views=16`, train on 4-view sequences with random view dropout `p_drop_view ∈ {0.0, 0.25, 0.5}`, and evaluate at `V ∈ {2, 3, 4, 5, 6}` by subsampling the same validation rig. Compare MPJPE against the fixed 4-view baseline.
2. **Cross-dataset generalization.** Train on H36M/WebBridge (4 views) with view dropout, then evaluate zero-shot on AIST++ (9 views) and MPI-INF-3DHP (14 views) to test `max_n_views` transfer.
3. **View-failure robustness.** Drop each view independently at inference and measure `MPJPE@V_remaining`.
4. **Ablation of positional embeddings.** Compare learned view embeddings vs. sinusoidal vs. camera-ray embeddings to avoid view-ID overfitting.

## Metrics to Track

- MPJPE/PA-MPJPE per view count (`MPJPE@V=2,3,4,...`).
- PCK@50/100/150 mm and AUC per view count.
- `ΔMPJPE`: degradation relative to the fixed 4-view baseline.
- Uncertainty calibration by view count (predicted log-variance should rise when fewer views are present).
- Failure rate: fraction of samples where `V < 2` or DLT becomes singular.

## Risks and Mitigations

- **`max_n_views` too small for 9–14 view datasets.** Set `max_n_views=16` or `32`; memory cost is only `O(max_n_views * d)`.
- **Positional embeddings overfit to training view IDs.** Use ray-direction or sinusoidal embeddings, or drop view IDs during training.
- **Mixed-batch padding increases memory.** Implement dynamic padding and gradient accumulation.
- **`V=2` triangulation is noisy.** Add a view-count auxiliary loss and a single-view shape prior in later iterations.
- **Refactoring all wrappers risks regressions.** Keep old fixed-view classes and add `VariableView*` variants side-by-side.
- **Dataset converters need view-subsampling.** Add a generic `subsample_views(npz, V)` helper in `motionflow_mv/data/`.

## Immediate Action Plan

1. **Week 1**: Create `RayAttentionFusionModelTemporalCrossviewVariableV1` with `max_n_views`, sliced embeddings, and view masking. Keep existing fixed-view models unchanged.
2. **Week 1**: Implement `subsample_views` data helper and a mixed-view `collate_fn`.
3. **Week 2**: Add view-dropout augmentation and train on MPI-INF-3DHP.
4. **Week 3**: Run the variable-view evaluation protocol and a cross-dataset smoke test on AIST++.
5. **Week 4**: Back-port the variable-view design to the combined uncertainty+residual+learned-triangulation model if results justify it.

## Conclusion

Variable view count is not a radical architectural change; the attention and triangulation heads are already view-permutation invariant. The work is removing hard-coded `n_views` assumptions, adding a view mask, and training with view dropout to improve real-world robustness and the paper narrative.
