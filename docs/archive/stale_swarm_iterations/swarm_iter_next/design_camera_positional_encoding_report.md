# Design: Camera Positional Encoding for Variable-View Fusion

## 1. Motivation

The current ray-attention temporal/cross-view models (`RayAttentionFusionModelTemporalCrossview` and `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`) add a learned `view_pos_embed` of size `(n_views, d)` to the `(time, view, joint)` tokens. This embedding is dataset-specific: it fixes the number of cameras, assumes views are ordered by an arbitrary ID, and carries no geometric meaning. On MPI-INF-3DHP (14 views) the model must be trained with `n_views=14`; on Human3.6M (4 views) the embedding does not transfer. A geometry-based camera positional encoding removes these constraints.

## 2. Design decisions

**Geometry-derived tokens.** For each view `v`, compute from calibrated cameras:

```
c_v = -R_v^T t_v                 # camera center in world
r_v = R_v^T [0, 0, 1]^T          # principal ray direction in world
f_v = (f_x + f_y) / 2            # mean focal length
```

**Scale normalization.** Make the encoding invariant to absolute scene scale and focal length:

```
\bar{c}   = mean_v c_v
s         = max_v ||c_v - \bar{c}||
\tilde{c}_v = (c_v - \bar{c}) / (s + eps)
\tilde{f}_v = f_v / mean_w f_w
```

**Fourier positional encoding.** Encode each scalar component with multiple sinusoidal bands, analogous to NeRF-style \gamma:

```
\gamma(p) = [p, sin(2^0 \pi p), cos(2^0 \pi p), ..., sin(2^{L-1} \pi p), cos(2^{L-1} \pi p)]
```

The final per-view camera position token is:

```
e_v = MLP([ \gamma(\tilde{c}_v), \gamma(r_v), \gamma(\tilde{f}_v) ])  in R^d
```

**Injection point.** Add `e_v` to the spatio-temporal tokens before the `(time, view)` transformer, replacing `view_pos_embed`. Because `e_v` depends only on physical camera geometry, the same model can accept any number of views and generalize across camera rigs.

## 3. Expected benefits

- **Variable view count:** no fixed `n_views`; slice or pad by actual `V`.
- **Cross-dataset transfer:** encoding is invariant to absolute scene scale and image resolution.
- **Stronger inductive bias:** view tokens carry camera location and viewing direction rather than memorized IDs.

## 4. References to existing files

- `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py` lines 88-89, 192-195: learned `view_pos_embed` and injection.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py` lines 245-249: same learned `view_pos_embed` in the current best model.
- `motionflow_mv/fusion/ray_attention_v4_model.py` lines 42-77: normalized camera feature extraction (`_normalized_camera_embedding`).
- `docs/swarm_iter11_variable_view_count_report.md`: motivation for variable-view support and `view_pos_embed` bottleneck.
