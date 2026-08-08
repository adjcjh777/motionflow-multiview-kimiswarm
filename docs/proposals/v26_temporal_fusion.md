# v25: Temporal Fusion Extension for v18 Deformable Cross-View Attention

**Task identifier:** `design_v25_temporal_fusion`  
**Depends on:** v18 (`docs/proposals/v18_deformable_attention.md`), v19 (`docs/proposals/v19_temporal_perceiver.md`)  

## Motivation

v18 introduced **DeformableCrossViewAttention** — a sparse, epipolar-geometry-guided attention block that reduces cross-view attention from `O(V²)` to `O(V · n_samples)`. It currently reasons **within a single frame**. The open question in the v18 proposal asks whether the sampler can be extended to attend to **temporal neighbours as well**.

We propose **v25: Deformable Spatio-Temporal Cross-View Attention**, a direct temporal fusion extension of v18. Instead of refining each frame in isolation, v25 lets every `(time, view, joint)` query token attend to a small, learned set of **key view × temporal offset** pairs. This adds explicit temporal reasoning **before triangulation**, at the same sparse cost as v18, and keeps the rest of the v5/v18 pipeline unchanged.

Key differences from existing temporal components:

| Component | Level | What it fuses | Cost |
|-----------|-------|---------------|------|
| v19 Temporal Perceiver | Post-triangulation 3D poses | Whole clip → latent set | `O(T · J · n_latents)` |
| v25 Deformable Spatio-Temporal Attention | Feature tokens before triangulation | Sparse view+time samples per token | `O(T · V · n_samples · n_temporal)` |
| ST transformer in v5 | Feature tokens | All time×view tokens | `O((T·V)²)` |

v25 is therefore a **lightweight middle ground**: it gives the model a temporal receptive field without the quadratic memory of the full ST transformer or the large latent set of the v19 Perceiver.

## Design

### Module

File: `motionflow_mv/fusion/deformable_spatio_temporal_attention_v25.py`

```text
DeformableSpatioTemporalAttention(
    d: int,
    n_heads: int = 4,
    n_views: int = 4,
    n_samples_view: int = 2,
    n_temporal_offsets: int = 3,        # [-1, 0, +1] by default
    epipolar_temperature: float = 10.0,
    motion_temperature: float = 1.0,
    dropout: float = 0.0,
)
```

### Inputs / outputs

- **Input tokens**: `(B, T, V, J, d)` — same layout as v18.
- **Camera parameters**: same as v18, shapes `(B*T, V, 3, 3)` for `K`, `R`, and `(B*T, V, 3)` for `t`.
- **Points**: `points_2d` `(B*T, V, J, 2)` used for epipolar distances.
- **Optional view mask**: `(B, T, V)` or `(B, V)`.
- **Output**: `(B, T, V, J, d)` updated tokens, same layout.

### Mechanism

For each query token at frame `t_q` and view `v_q`, the module builds a small candidate set of key tokens:

1. **Temporal offsets** — a fixed learned set of integer offsets `Δt ∈ [-k, …, +k]` (default `k=1`).
   - Only valid frames are kept (clamped at clip boundaries).
   - Each offset is associated with a learnable positional embedding added to the key token.

2. **Content logits** — same scaled dot-product as v18, but over the `(temporal offset, key view)` candidate set:
   ```
   logits_content[h, t_q, v_q, Δt, v_k, j] = (Q[t_q, v_q, j, h] · K[t_q+Δt, v_k, j, h]) / sqrt(Dh)
   ```

3. **Geometry logits** — two additive terms:
   - **Epipolar term** for the view pair `(v_q, v_k)` at the query frame `t_q` (reused from v18).
   - **Temporal motion term** that rewards 3D-consistent motion between `t_q` and `t_q+Δt`. We approximate it with the inverse of the per-joint reprojection residual variance across views, which is already computed in the v5 pipeline.
   ```
   logits_geom = -epi_dist[v_q, v_k, j] * softplus(geom_scale) / epipolar_temperature
                 - motion_cost[t_q, Δt, j] / motion_temperature
   ```

4. **Sparse top-k sampling** — softmax over the candidate set, then keep only the top-`n_samples_view` key views **for each temporal offset**. Gradients flow through the same straight-through estimator used in v18.

5. **Aggregation** — values are aggregated with re-normalised sparse weights, projected back to `d` dimensions, and a gated residual + layer-norm is applied.

6. **View and temporal masking** — masked views are excluded, and out-of-bound temporal offsets are ignored.

### Complexity

- Number of candidates per query: `n_temporal_offsets × V`.
- Effective attention cost: `O(B · T · V · J · n_samples_view · n_temporal_offsets)` — linear in `T` and `V`, with the constant determined by the small sparsity parameters.
- Memory: only the candidate logits `(B*T, V, n_temporal_offsets, V, J)` need to be materialised; for `V=4` and `n_temporal_offsets=3` this is negligible compared to the full `(T·V)²` ST transformer.

## Integration with `OmniMultiViewFusionV5`

v25 is designed to be **drop-in next to v18**. Add a new toggle:

```python
use_deformable_spatio_temporal_attention_v25: bool = False,
```

Instantiate in `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
self.use_deformable_spatio_temporal_attention_v25 = use_deformable_spatio_temporal_attention_v25
if self.use_deformable_spatio_temporal_attention_v25:
    self.deformable_spatio_temporal_attention_v25 = DeformableSpatioTemporalAttention(
        d=d,
        n_heads=n_heads,
        n_views=n_views,
        n_samples_view=max(2, n_views // 2),
        n_temporal_offsets=3,
        epipolar_temperature=10.0,
        motion_temperature=1.0,
        dropout=0.1,
    )
```

Hook into the forward pass immediately **after** the v18 deformable cross-view attention block:

```python
# v18 sparse cross-view attention (per-frame).
if self.use_deformable_cross_view_attention_v18 and self.deformable_cross_view_attention_v18 is not None:
    feat = self.deformable_cross_view_attention_v18(
        feat, K=K_corrected, R=R, t=t, points_2d=points_2d, view_mask=view_mask
    )

# v25 sparse spatio-temporal attention.
if self.use_deformable_spatio_temporal_attention_v25 and self.deformable_spatio_temporal_attention_v25 is not None:
    feat = self.deformable_spatio_temporal_attention_v25(
        feat, K=K_corrected, R=R, t=t, points_2d=points_2d, view_mask=view_mask
    )
```

This ordering lets v18 first refine per-frame cross-view features, and v25 then diffuse temporal information at the same sparse cost.

## Configuration & hyperparameters

Suggested defaults for a first smoke test:

| Hyperparameter | Default | Rationale |
|---------------|---------|-----------|
| `n_samples_view` | `max(2, n_views // 2)` | Same as v18; keeps cross-view sparsity. |
| `n_temporal_offsets` | `3` | Covers `[-1, 0, +1]`; small enough to stay cheap. |
| `epipolar_temperature` | `10.0` | Matches v18. |
| `motion_temperature` | `1.0` | Start with equal weighting; tune if temporal smoothing is too strong. |
| `dropout` | `0.1` | Match v18/ST transformer. |

## Training considerations

- **Warm start**: the residual gate is initialised to ~0, so enabling v25 on a v18 checkpoint is safe and starts as identity.
- **Stacking order**: keep v18 first, then v25, then the existing ST transformer. v25 provides local temporal context; the ST transformer still provides global context when memory allows.
- **Losses**: no new losses are required. Standard 3D pose supervision and the existing `epi_loss` drive the content and geometry terms.
- **Clip length**: because attention is linear in `T`, v25 works for both short (`T=9`) and long (`T=256`) clips without special handling.
- **Relation to v19 / v22 / v23 / v24**:
  - v19 (Temporal Perceiver) can remain as a final post-triangulation refiner.
  - v22 (Kinematic Anthropometric Prior) is unaffected; it still operates on the per-frame refined 3D pose.
  - v23/v24 stack on v18 + KAP (± fixed BA). v25 is orthogonal and can be combined with them to form **v25 = v18 + KAP + deformable spatio-temporal attention** (or v18 + fixed BA + KAP + v25).

## Test coverage

Add `tests/test_deformable_spatio_temporal_attention_v25.py` covering:

- Output shape matches input shape `(B, T, V, J, d)`.
- Gradients reach input and parameters.
- Masked views are zeroed.
- Temporal boundary handling (`T < n_temporal_offsets` and boundary frames).
- `n_temporal_offsets=1` (identity/no-op) and `n_temporal_offsets=3`.
- Invalid head dimensions raise `ValueError`.

Run:

```bash
pytest tests/test_deformable_spatio_temporal_attention_v25.py -q
```

Also add a toggle-on case to `tests/test_omniview_fusion_v5.py` to ensure `use_deformable_spatio_temporal_attention_v25=True` integrates cleanly with the full v5 forward pass.

## Open questions / future work

1. **Learned temporal offsets**: instead of fixed `[-k, …, +k]`, learn a small set of continuous temporal offsets and sample them with linear interpolation, similar to deformable image transformers.
2. **Motion prior**: replace the reprojection-residual motion cost with an explicit 3D velocity/trajectory smoothness term.
3. **Multi-scale temporal sampling**: use different temporal granularities (e.g. short and long offsets) in separate heads.
4. **Replace ST transformer**: if v25 proves sufficient, consider disabling the full ST transformer to save memory and rely on v18 + v25 for spatio-temporal reasoning.

## References

- `motionflow_mv/fusion/deformable_cross_view_attention.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `docs/proposals/v18_deformable_attention.md`
- `docs/proposals/v19_temporal_perceiver.md`
- `docs/proposals/v22_kinematic_anthropometric_prior.md`
