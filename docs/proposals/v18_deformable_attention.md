# V18: Deformable Cross-View Attention

## Motivation

Existing cross-view transformers in `OmniMultiViewFusionV5` attend to **all**
views at once.  When the number of views grows, this becomes expensive and
mixes in views that are geometrically irrelevant for a given joint.  Deformable
attention (DETR-style) shows that attending to a small, learned subset of
positions is sufficient if the sampler is guided by a good geometric prior.

We propose a **deformable cross-view attention block** that uses **epipolar
geometry** as the prior and learns to refine a sparse set of sampled key views
per query view/joint.

## Design

### Module

File: `motionflow_mv/fusion/deformable_cross_view_attention.py`

```text
DeformableCrossViewAttention(
    d: int,
    n_heads: int = 4,
    n_views: int = 4,
    n_samples: int = 2,
    epipolar_temperature: float = 10.0,
    dropout: float = 0.0,
)
```

### Inputs / outputs

- **Input tokens**: `(B, T, V, J, d)` per-view per-joint features.
- **Camera parameters**:
  - `K`: `(B*T, V, 3, 3)` intrinsics
  - `R`: `(B*T, V, 3, 3)` rotations
  - `t`: `(B*T, V, 3)` translation
- **Points**: `points_2d` `(B*T, V, J, 2)` used to compute epipolar lines.
- **Optional view mask**: `(B, T, V)` or `(B, V)` marking present views.
- **Output**: `(B, T, V, J, d)` updated tokens, same layout.

### Mechanism

1. **Project Q, K, V** from per-view tokens with a single linear layer and split
   into multi-head tensors of shape `(N, V, J, H, Dh)`.

2. **Content logits** are standard scaled dot products across views:
   ```
   logits_content[h, q, k, j] = (Q[q,j,h] · K[k,j,h]) / sqrt(Dh)
   ```

3. **Geometry logits** come from pairwise epipolar distances between the query
   view `q` and key view `k` for joint `j`.  Smaller epipolar distance should
   produce a larger logit:
   ```
   logits_geom = -dist[q, k, j] * softplus(geometry_scale) / epipolar_temperature
   ```

4. **Sparse top-k sampling**.  We compute soft attention weights with softmax,
   then in the **forward pass** keep only the top-`n_samples` key views per
   query.  Gradients flow through a straight-through estimator so the module
   remains trainable end-to-end.

5. **Aggregation**.  Values are aggregated with the re-normalized sparse
   weights, projected back to `d` dimensions, and a residual + layer-norm is
   applied.

6. **View masking**.  Masked key views are excluded from attention, and masked
   query views are zeroed in the output.

## Why this fits the project

- **Epipolar-aware**: directly uses calibrated geometry, consistent with the
  existing `epipolar_bias` and `epipolar_transformer_bias` machinery.
- **Sparse / efficient**: reduces attention from `O(V^2)` to `O(V * n_samples)`.
- **Variable views**: the top-k operation is defined on the fly and therefore
  handles any number of available views.
- **End-to-end differentiable**: straight-through estimator allows gradient
  descent to learn the sampling.

## Integration plan

1. Instantiate inside `OmniMultiViewFusionV5` right after the per-view feature
   extraction and before the spatio-temporal transformer.
2. Add a configuration flag, e.g. `use_deformable_cross_view_attention=True`.
3. Provide `n_samples` as a hyperparameter; start with `n_samples=2` for `V=4`.
4. Train a smoke experiment and compare MPJPE / runtime against the full
   cross-view transformer baseline.

## Test coverage

`tests/test_deformable_cross_view_attention.py` verifies:

- Output shape matches input shape.
- Gradients reach input and parameters.
- Masked views are zeroed.
- Edge cases `n_samples=1` and 2D view masks work.
- Invalid head dimensions raise `ValueError`.

## Open questions / future work

- Should `n_samples` be learned per joint instead of global?
- Can we extend the sampler to attend to **temporal** neighbours as well, i.e.
  deformable spatio-temporal sampling?
- Should the geometry term also include camera ray directions / baseline angle?
