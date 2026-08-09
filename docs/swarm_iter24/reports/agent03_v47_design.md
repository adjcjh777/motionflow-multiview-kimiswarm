# Agent-03 — TemporalAggregationV47 API & Integration Design

**Owner:** Agent-03 (DESIGN)  
**Branch:** `v47-temporal`  
**Tracking issue:** #162  
**Status:** Design report — ready for Agent-04 implementation  

## Scope

This report finalizes the public API, internal architecture, and integration plan for `TemporalAggregationV47`.  It is derived from `docs/proposals/v47_combined_architecture.md` and is written against the current state of `v47-temporal` (commit pulled at task start).  No source files are modified in this DESIGN task.

## Design goals

1. **Lightweight post-triangulation smoother.** v47 sits *after* the per-frame v25/v45/v46 triangulation.  It does not replace geometry fusion.
2. **Identity at initialization.** With the residual gate initialized to `0.0`, the module is a no-op at the start of training, which protects the v46 baseline during warm-up.
3. **Variable views and variable-length clips.** The module must accept `view_mask` (which views are present per frame) and `clip_mask` (which frames are valid).
4. **No duplication of v26/v35 temporal code.** v26 operates on mid-level ray/feature tokens; v35 operates on the (view, joint) graph.  v47 operates only on the final 3D pose trajectory `(B, T, J, 3)` and is therefore a separate, minimal head.

## Final module API

**File:** `motionflow_mv/fusion/temporal_aggregation_v47.py`

```python
from typing import Optional

import torch
import torch.nn as nn


class TemporalAggregationV47(nn.Module):
    """Lightweight temporal refinement head for sparse-view 3D pose trajectories.

    The module flattens the (time, joint) dimensions into a single token sequence,
    applies a small transformer encoder, and returns a gated residual update to
    the input 3D poses.  At initialization the residual gate is zero, so the
    module is an identity mapping.

    Parameters
    ----------
    n_joints:
        Number of skeleton joints ``J``.
    d_model:
        Hidden dimension of the temporal transformer.
    n_heads:
        Number of self-attention heads.
    num_layers:
        Number of transformer encoder layers.
    temporal_window:
        If ``None``, attention is global over the clip.  If an ``int`` is given,
        each frame only attends to the ``window // 2`` frames on either side
        (causal-friendly for future streaming work).
    dropout:
        Dropout probability inside the transformer layers.
    residual_gate_init:
        Initial value of the learnable scalar gate ``g``.  Set to ``0.0`` so the
        residual path is disabled at start.
    use_view_count_conditioning:
        If ``True``, each (time, joint) token is concatenated with the scalar
        ``log(max(n_views_t, 1))`` so the temporal head knows how much to trust
        each frame's per-frame estimate.
    """

    def __init__(
        self,
        n_joints: int = 17,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        temporal_window: Optional[int] = None,
        dropout: float = 0.1,
        residual_gate_init: float = 0.0,
        use_view_count_conditioning: bool = True,
    ) -> None:
        ...

    def forward(
        self,
        poses_3d: torch.Tensor,               # (B, T, J, 3)
        view_mask: torch.Tensor,              # (B, T, V)
        clip_mask: Optional[torch.Tensor] = None,  # (B, T) True = valid frame
    ) -> torch.Tensor:
        """Return temporally refined 3D poses of shape (B, T, J, 3)."""
        ...
```

### Tensor shapes and conventions

- `poses_3d`: `(B, T, J, 3)` — per-frame triangulated poses from v46.
- `view_mask`: `(B, T, V)` — binary mask indicating which views contributed to each frame.  This is the same `view_mask` used by `SparseViewGeneralizationV46.forward`.
- `clip_mask`: `(B, T)` — boolean tensor.  ``True`` / ``1`` means the frame is valid.  Padding frames should be set to ``False`` / ``0``.
- Return: `(B, T, J, 3)` — refined poses.

### Internal architecture

```text
Input: poses_3d (B, T, J, 3)
        |
        ▼
[ Optionally concat log(n_views_t) per frame ]
        |
        
Reshape to tokens (B, T*J, 3[+1])
        |
        ▼
Linear projection to d_model
        |
        ▼
Add learned (time, joint) positional embeddings  # shared for all J*J positions or per position
        |
        ▼
num_layers × TransformerEncoderLayer (self-attention over T*J tokens)
        |
        ▼
Linear projection back to 3
        |
        ▼
Reshape to (B, T, J, 3) residual ΔP_t
        |
        ▼
poses_3d + sigmoid(g) * ΔP_t
```

1. **Token layout.**  Flatten `(T, J)` into `N = T * J` tokens.  Each token carries its 3D coordinate.  The attention mask blocks invalid tokens (from `clip_mask`) from attending to or being attended by any other token.
2. **Positional encoding.** Use a learned `(time, joint)` embedding of shape `(max_temporal_len, J, d_model)`, or a sinusoidal encoding if learned parameters are undesirable.  Either is acceptable; the implementation should reuse the existing `max_temporal_len` convention from `OmniMultiViewFusionV5`.
3. **View-count conditioning (optional).** Compute `v_t = log(max(n_views_t, 1))` for each frame, broadcast to `(B, T, J, 1)`, and concatenate to the 3D coordinates before the first linear projection.  This gives the transformer a frame-level confidence cue without requiring a separate view-reliability tensor.
4. **Windowed attention.** If `temporal_window` is set, build a band-diagonal attention mask so token `(t, j)` only attends to tokens within `[t - w//2, t + w//2]`.  This keeps memory and latency bounded and is compatible with future streaming inference.
5. **Residual gate.** A single learnable scalar `g` initialized to `residual_gate_init` (default `0.0`).  The output is `poses_3d + sigmoid(g) * delta`.  This is the same warm-start idiom used by v26 and v33 ray attention.

### Default hyperparameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `n_joints` | `17` | H36M skeleton; 28 for MPI-INF-3DHP via `j` argument. |
| `d_model` | `64` | Matches the v5 base dimension `d`. |
| `n_heads` | `4` | Consistent with most v5 transformer heads. |
| `num_layers` | `2` | Shallow; avoids over-smoothing fast motion. |
| `temporal_window` | `None` | Full-clip attention by default; set to `7` for streaming. |
| `dropout` | `0.1` | Standard v5 dropout. |
| `residual_gate_init` | `0.0` | Identity at init. |
| `use_view_count_conditioning` | `True` | Required for sparse-view robustness. |

## Integration into `OmniMultiViewFusionV5`

### 1. Constructor changes

In `motionflow_mv/fusion/omniview_fusion_v5.py`, add to `__init__` after the v46 block (around line 773):

```python
# Optional v47 temporal aggregation over sparse-view triangulated poses.
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
        residual_gate_init=0.0,
        use_view_count_conditioning=v47_use_view_count_conditioning,
    )
else:
    self.temporal_aggregation_v47 = None
```

Add the following parameters to the `__init__` signature of `OmniMultiViewFusionV5`:

```python
use_v47_temporal_aggregation: bool = False,
v47_temporal_d_model: int = 64,
v47_temporal_n_heads: int = 4,
v47_temporal_num_layers: int = 2,
v47_temporal_window: Optional[int] = None,
v47_temporal_dropout: float = 0.1,
v47_temporal_loss_weight: float = 0.01,
v47_use_view_count_conditioning: bool = True,
```

### 2. Forward pass changes

The v47 head should be invoked **after** the v46 sparse-view triangulation has produced per-frame 3D poses and **before** the final pose is returned to the trainer.  The exact insertion point depends on the existing v46 wiring; the convention is to place it immediately after the line where `poses_3d` / `P_t` becomes available with shape `(B, T, J, 3)`.

Pseudo-code:

```python
if self.use_v47_temporal_aggregation:
    poses_3d = self.temporal_aggregation_v47(
        poses_3d=poses_3d,
        view_mask=view_mask,
        clip_mask=clip_mask,  # optional; if None, all frames valid
    )
```

Where `view_mask` is the same `(B, T, V)` tensor already normalized by `_prepare_view_mask`.  If the current forward only has a flattened `(B*T, V)` version, reshape it back to `(B, T, V)` before calling v47.

### 3. Training-loop changes

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

1. Add CLI flags mirroring the constructor arguments:

```python
parser.add_argument("--use_v47_temporal_aggregation", action="store_true")
parser.add_argument("--v47_temporal_d_model", type=int, default=64)
parser.add_argument("--v47_temporal_n_heads", type=int, default=4)
parser.add_argument("--v47_temporal_num_layers", type=int, default=2)
parser.add_argument("--v47_temporal_window", type=int, default=None)
parser.add_argument("--v47_temporal_dropout", type=float, default=0.1)
parser.add_argument("--v47_temporal_loss_weight", type=float, default=0.01)
parser.add_argument("--v47_use_view_count_conditioning", action="store_true", default=True)
```

2. Pass the flags through `build_model_from_args` to `OmniMultiViewFusionV5`.

3. Add a temporal smoothness loss only when `use_v47_temporal_aggregation` is enabled:

```python
if args.use_v47_temporal_aggregation and args.v47_temporal_loss_weight > 0:
    temporal_loss = (
        args.v47_temporal_loss_weight
        * (poses_3d[:, 1:] - poses_3d[:, :-1]).abs().mean()
    )
    loss = loss + temporal_loss
```

Note: the loss is computed on the **refined** output.  The v47 module itself does not own the loss; the trainer applies it.

### 4. Staged training recipe

The proposal recommends:

1. Load a trained v46-SVG checkpoint (or train v46 from scratch).
2. For the first epoch, freeze all parameters except the v47 head so it learns on stable per-frame estimates.
3. Unfreeze all layers and fine-tune end-to-end.

This is a trainer concern, not a model concern.  The trainer can implement it by iterating over parameter groups after model construction:

```python
if args.use_v47_temporal_aggregation and args.v47_freeze_base_epochs > 0:
    for name, param in model.named_parameters():
        if "temporal_aggregation_v47" not in name:
            param.requires_grad = False
```

A dedicated flag `--v47_freeze_base_epochs` (default `1`) is recommended but is part of the trainer task, not the module design.

## Mask handling details

### View mask → view count

Inside `TemporalAggregationV47.forward`, derive per-frame view counts:

```python
view_counts = view_mask.sum(dim=-1)  # (B, T)
```

If `use_view_count_conditioning` is enabled, use `log(view_counts.clamp(min=1))` and broadcast to shape `(B, T, 1)` before concatenation.

### Clip mask → attention mask

If `clip_mask` is `None`, treat all frames as valid.  Otherwise:

```python
# token_valid: (B, T)
token_valid = clip_mask  # True = valid
# Build additive attention mask for flattened (T, J) tokens.
B, T, J = poses_3d.shape
valid_flat = token_valid.unsqueeze(-1).expand(-1, -1, J).reshape(B, T * J)
attn_mask = valid_flat.unsqueeze(2) * valid_flat.unsqueeze(1)  # (B, T*J, T*J)
attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float("-inf"))
```

For `nn.TransformerEncoder` with `batch_first=True`, pass the additive mask via the `src_mask` or `mask` argument depending on the exact layer wrapper.

### Windowed attention

When `temporal_window` is set, compute a `(T, T)` boolean mask `window_mask[t, s] = abs(t - s) <= window // 2`, broadcast over joints, and combine with the clip mask via element-wise `&` before converting to the additive mask.

## Relationship to existing temporal modules

| Module | Operates on | Role | v47 interaction |
|--------|------------|------|-----------------|
| `temporal_geometry_fusion_v26.py` | Mid-level ray/feature tokens | Geometry-aware temporal fusion inside v25 | Independent; v47 runs after triangulation. |
| `temporal_view_joint_graph_network_v35.py` | (view, joint) graph tokens | Temporal edges across frames in graph network | Independent; v47 is pose-level only. |
| `temporal_aggregation_v47.py` | Final 3D pose `(B, T, J, 3)` | Post-triangulation temporal smoother | New module; no shared code. |

The key differentiator is that v47 does **not** touch feature tokens, rays, or view graphs.  This keeps it orthogonal to v26/v35 and avoids duplicating their temporal logic.

## Unit test contract

The implementation must satisfy the tests that will be written in `tests/test_temporal_aggregation_v47.py`:

1. **Shape preservation.** Output shape equals input shape `(B, T, J, 3)`.
2. **Identity at init.** With the default residual gate init, the module output is approximately equal to the input for random data.
3. **Mask correctness.**
   - Tokens from masked-out frames do not affect valid tokens.
   - Output for masked frames can be zeroed or left unchanged; the contract is that valid-frame outputs are unaffected.
4. **Gradient flow.** Backward pass produces non-`None` gradients for all v47 parameters.
5. **Windowing.** When `temporal_window=3`, a token at frame `t` has nonzero attention weights only within one frame of `t`.
6. **View-count conditioning.** With `use_view_count_conditioning=True`, the module accepts and uses the `view_mask` without error.

## Success criteria

- API matches the contract above.
- `TemporalAggregationV47` can be imported and instantiated with all defaults.
- `OmniMultiViewFusionV5` loads with `use_v47_temporal_aggregation=True` and no shape errors.
- Smoke config `configs/benchmark_v47_temporal_svg_smoke.yaml` runs end-to-end on RTX 4090.
- Evaluation in `experiments/eval_variable_views.py` reports `MPJPE@k` for v46 vs v47 and shows ≥5% improvement at `k ≤ 3`.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Over-smoothing fast motion | Keep `num_layers=2`, gate init `0.0`, and tune `v47_temporal_loss_weight`. |
| OOM with full-clip attention | Default `temporal_window=None` is fine for short clips; use `temporal_window=7` for long clips or streaming. |
| v46 base not yet ready | v47 implementation can proceed; smoke comparison must wait for v46 smoke results. |
| Duplicating v26/v35 logic | Restrict v47 to pose-level tokens only, as specified above. |

## References

- `docs/proposals/v47_combined_architecture.md` — original proposal
- `docs/swarm_iter24_action_plan.md` — agent task assignments
- `motionflow_mv/fusion/sparse_view_generalization_v46.py` — v46 base
- `motionflow_mv/fusion/omniview_fusion_v5.py` — insertion point for v47
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — trainer wiring
