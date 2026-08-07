# Cross-View Transformer v17

## Motivation

``VariableViewSetAggregator`` is permutation-invariant and therefore agnostic
to the actual geometric arrangement of the cameras.  For calibrated multi-view
setups the relative camera geometry is known, and a transformer that uses this
information can learn more discriminative cross-view attention patterns.  This
design document proposes ``CrossViewTransformerV17``.

## What it does

``CrossViewTransformerV17`` replaces (or augments) the view-set aggregator in
``OmniMultiViewFusionV5`` with a stack of cross-view transformer layers.  Each
view token is tagged with a **ray/camera embedding** computed from the camera
intrinsics/extrinsics and the 2D keypoint, so attention scores can depend on
viewing rays and camera centres.

## Inputs / outputs

- **Input features**: ``(B, T, V, J, d)``
- **Inputs (optional)**: ``K (B*T, V, 3, 3)``, ``R (B*T, V, 3, 3)``,
  ``t (B*T, V, 3)``, ``points_2d (B, T, V, J, 2)``, ``view_mask``
- **Output**: ``(B, T, V, J, d)`` (same shape, masked views zeroed)

## Ray/camera embedding

For each view ``v`` and joint ``j``:

1. Convert the 2D point to a ray direction in camera coordinates:
   ``d_cam = K_v^{-1} [u, v, 1]^T``.
2. Rotate to world coordinates: ``d_world = R_v^T d_cam`` and normalise.
3. Compute camera centre ``c_v = -R_v^T t_v``.
4. Concatenate ``[d_world, c_v]`` (6-D) and linearly project to ``d``.

This embedding is added to queries and keys inside each transformer layer.

## Architecture

- ``n_layers`` stacked ``_CrossViewTransformerLayer`` blocks.
- Each block contains:
  - Separate Q/K/V linear projections.
  - ``nn.MultiheadAttention`` with key-padding mask from ``view_mask``.
  - Residual connection + layer norm.
  - Feed-forward network (4d -> d) with GELU.

## Integration into OmniMultiViewFusionV5

In ``omniview_fusion_v5.py``, after the view embedding block and before the
spatio-temporal transformer:

```python
if self.use_cross_view_transformer:
    feat = self.cross_view_transformer(
        feat,
        K=K_corrected,
        R=R,
        t=t,
        points_2d=points_2d,
        view_mask=view_mask,
    )
```

Because the module preserves shape, it can be dropped in place of
``VariableViewSetAggregator``.

## Test coverage

- Forward shape preservation.
- No-camera fallback.
- View masking (masked views are zeroed).
- Gradient flow.

## Future work

- Add an optional epipolar attention bias (distance of a ray to the epipolar
  line of another view) inside the attention scores.
- Add learned positional embeddings per view to recover some permutation
  invariance when camera parameters are unavailable.
- Ablate against ``VariableViewSetAggregator`` on H36M and MPIInf3DHP.
