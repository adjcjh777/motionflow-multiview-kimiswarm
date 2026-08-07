# v19 Temporal Perceiver Refiner

## Motivation

Long video clips are currently processed by the v5 model in two ways:

1. **Sliding-window temporal refinement** (`temporal_refiner.py`) — a small GRU over a short window of DLT triangulated poses, borrowing information from neighbouring frames.
2. **Time+view transformer** inside `omniview_fusion_v5.py` — a joint attention over time and views, but it scales quadratically with temporal length and view count.

For clips with hundreds of frames, the time+view transformer becomes memory-intensive, and the GRU window is too short to capture long-range temporal relations.  The **Temporal Perceiver** addresses both limitations by compressing the entire clip to a small fixed-size latent set and decoding per-frame 3D poses from that latent set, giving **linear memory complexity in clip length** and a **global temporal receptive field**.

## Module

File: `motionflow_mv/fusion/temporal_perceiver_v19.py`

```python
class TemporalPerceiverRefiner(nn.Module):
    def __init__(
        self,
        j: int = 17,
        in_dim: int = 3,
        d: int = 64,
        n_latents: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        max_temporal_len: int = 256,
    ): ...
```

### Inputs

- `x`: `(B, T, J, in_dim)` — per-frame per-joint feature tokens.  Typical inputs include:
  - The raw per-frame triangulated 3D pose (`in_dim = 3`)
  - The concatenation of the triangulated pose with per-view pooled features
  - The output of the v5 residual MLP before decoding
- `baseline_3d`: optional `(B, T, J, 3)` baseline pose.  The module predicts a residual and adds it to the baseline.  If omitted, the raw residual is returned.

### Outputs

- `refined_3d`: `(B, T, J, 3)` — one refined 3D pose per frame.

### Architecture

1. **Projection + positional embeddings**
   - `input_proj`: linear from `in_dim` to `d`.
   - Add learned time embeddings `(T, d)` and joint embeddings `(J, d)`.
   - Flatten to `(B, T*J, d)`.

2. **Perceiver encoder**
   - A fixed set of `n_latents` latent vectors summarises the whole temporal clip.
   - Each encoder layer performs:
     - latent self-attention,
     - input self-attention over frame/joint tokens,
     - cross-attention from latents to input tokens,
     - feed-forward on the latents.
   - This is repeated for `n_layers` layers.

3. **Perceiver decoder**
   - Learned `(T, J, d)` query embeddings decode per-frame per-joint features from the latents.
   - Each decoder layer performs:
     - cross-attention from queries to latents,
     - feed-forward.
   - The same positional embeddings are added to the queries to preserve temporal and skeletal structure.

4. **Pose head**
   - A final linear layer maps `d` to 3 and produces the 3D residual.

### Complexity

- Encoder cross-attention: `O(B * n_latents * T * J)` — linear in temporal length.
- Decoder cross-attention: `O(B * T * J * n_latents)` — also linear in temporal length.
- Memory is dominated by the `(B, T*J, d)` input tensor and `(B, n_latents, d)` latent tensor.

## Integration with v5

The module is intentionally standalone so it can be inserted after the existing v5 forward pass without changing the core model:

```python
from motionflow_mv.fusion.temporal_perceiver_v19 import TemporalPerceiverRefiner

class OmniMultiViewFusionV5(...):
    def __init__(self, ..., use_temporal_perceiver: bool = False, ...):
        ...
        if use_temporal_perceiver:
            self.temporal_perceiver = TemporalPerceiverRefiner(
                j=j, in_dim=3, d=d, n_latents=32, n_layers=2
            )

    def forward(self, x, ...):
        # existing v5 logic: per-frame pose (B, T, J, 3)
        pred_3d = ...
        if self.use_temporal_perceiver:
            pred_3d = self.temporal_perceiver(pred_3d, pred_3d)
        return pred_3d, ...
```

A richer variant feeds the module the concatenation of the per-frame 3D pose and a pooled per-frame feature from the ST transformer, giving it both geometric and appearance/contextual cues.

## Test coverage

File: `tests/test_temporal_perceiver_v19.py`

- Shape check for forward pass with and without baseline.
- Gradient propagation test.
- Maximum temporal length smoke test (`T = 256`).

## Next steps / open questions

1. **Input features**: Should the perceiver receive only 3D poses, or also per-view visibility/uncertainty tokens?  Feeding visibility-aware features may help occluded frames.
2. **Hierarchical compression**: For very long clips (e.g. >500 frames), a two-stage perceiver with local window latents followed by global latents could be explored.
3. **Training objective**: Beyond 3D pose supervision, a temporal consistency loss or masked-joint prediction objective could improve generalisation.
4. **Integration experiment**: Train a v5 variant with `use_temporal_perceiver=True` on WebBridge and H36M to compare MPJPE against the sliding-window GRU baseline.
