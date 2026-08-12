# v52 Geometry-Aware Attention Pooling (GAAP)

## Motivation

OmniMultiViewFusionV5 fuses evidence through an ST transformer and per-joint
weight heads, but the fusion is still driven mainly by learned feature
correlations. The ST transformer applies an epipolar bias, yet it does not
explicitly pool geometrically-consistent rays before triangulation. Degenerate
or wide-baseline views can therefore remain over-weighted, especially after the
v45–v51 extensions add learned capacity. v52 **Geometry-Aware Attention Pooling
(GAAP)** introduces a geometry-conditioned cross-view attention module that sits
after the ST transformer and before the covariance/weight heads, letting the
model attend over views using camera rays and epipolar compatibility while
staying identity-at-init via a zero-initialized residual gate.

## Architecture

GAAP refines the ST-transformer feature tensor `feat` of shape
`(B, T, V, J, d)`.

1. **Ray embedding**. For each `(view, joint)` token compute the camera center
   `c_v = -R_v^T t_v` and the normalized viewing ray
   `ray_vj = normalize(R_v^T K_v^{-1} [u_vj, v_vj, 1]^T)`. A small MLP projects
   these to `d` channels and adds them to `feat`.

2. **Geometry-aware attention**. A transformer encoder layer processes view
tokens per `(B, T, J)` slice:

   ```
   Q, K, V = Linear(feat)                          # (B*T, V, J, d)
   geom_bias_vv' = MLP([c_v, c_v', ray_v, ray_v', angle(ray_v, ray_v')])
   logits_vv' = (Q_v K_v'^T)/sqrt(d/h) + geom_bias_vv' + mask_bias
   attn = softmax(logits)
   out_v = sum_v' attn_vv' V_v'
   ```

3. **Warm-start residual**. The output is projected back to `d` channels and
   added through a scalar gate initialized to `0.0`:

   ```
   feat' = feat + gate * MLP(out)
   ```

   With `gate = 0` the module is an identity at initialization, preserving any
   warm-started v45–v51 checkpoint.

If `v52_gaap_use_epipolar_pooling = True`, a Sampson-distance term is added to
`geom_bias` so geometrically-compatible views attract stronger attention.

## Inputs and Outputs

| Input | Shape | Description |
|-------|-------|-------------|
| `feat` | `(B, T, V, J, d)` | ST-transformer per-view per-joint features |
| `points_2d` | `(B, T, V, J, 2)` | Detected 2D keypoints |
| `K, R, t` | camera tensors | Intrinsics/extrinsics |
| `view_mask` | `(B, T, V)` | Binary view validity mask |

**Output:** `feat_refined` of shape `(B, T, V, J, d)`, which replaces the
original `feat` before the covariance and weight heads.

## Integration Point

Insert GAAP in `motionflow_mv/fusion/omniview_fusion_v5.py` immediately after the
ST transformer reshapes `feat` to `(B, T, V, J, d)` and before the covariance
head (around line 1513):

```python
if self.use_v52_geometry_aware_attention_pooling:
    feat = feat + self.geometry_aware_attention_pooling_v52(
        feat,
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        view_mask=view_mask_flat.view(B, T, V),
    )
```

Because the update is additive and gated to zero, configs that leave the toggle
off are unaffected.

## Config Flags

```yaml
use_v52_geometry_aware_attention_pooling: False
v52_gaap_hidden: 64
v52_gaap_n_heads: 4
v52_gaap_n_layers: 2
v52_gaap_dropout: 0.1
v52_gaap_residual_gate_init: 0.0        # identity at init
v52_gaap_use_epipolar_pooling: True
v52_gaap_epipolar_weight: 0.1
v52_gaap_use_view_count_conditioning: True
```

## Expected MPJPE Impact

- **Sparse views (V=2,3)**: 3–6 mm reduction by suppressing geometrically
  inconsistent views.
- **Full views (V=4+)**: 1–3 mm reduction through sharper per-joint view
  weighting on Human3.6M wide-baseline setups.
- **Cross-domain (WebBridge/MPI)**: 1–2 mm improvement because the geometry
  bias is dataset-agnostic.

These estimates assume GAAP is stacked on the strongest baseline (v45–v51) with
`v52_gaap_residual_gate_init = 0.0`.

## Risks (Summary)

1. Warm-start regression if the gate is not initialized to zero.
2. Epipolar bias can overwhelm useful learned correlations.
3. Variable view counts may break attention masking.
4. Extra `V×V` attention raises memory/latency.
5. Noisy camera parameters from mixed datasets can propagate errors.

See `docs/swarm_iter26/reports/agent_geometry_aware_attention_pooling_risks.md`
for detailed mitigations.

## Implementation Plan

1. Implement `motionflow_mv/fusion/geometry_aware_attention_pooling_v52.py` with
   ray embedding, geometry-biased attention, and a zero-initialized residual gate.
2. Wire the toggle and module into `OmniMultiViewFusionV5.__init__` and
   `forward()`, inserting it after the ST transformer.
3. Add a smoke YAML `configs/benchmark_v52_gaap_smoke.yaml` with a small sample
   count for fast RTX 4090 validation.
4. Run smoke and ablation against the v45–v51 baseline; tune
   `v52_gaap_epipolar_weight` and the residual gate schedule.
5. If smoke `val_MPJPE` is within 5 mm or better, queue a full A800 run via
   `scripts/launch_v33_a800_queue.py` and update `docs/swarm_iter26/status.md`.
