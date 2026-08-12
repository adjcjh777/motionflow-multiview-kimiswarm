# Agent-05 Design: v46 Sparse-View Generalization (SVG) Module API and Integration Notes

**Agent:** Agent-05 (DESIGN)  
**Branch:** `v46-svg`  
**Tracking issue:** #160  
**Depends on:** #154 (v25 all-train baseline), v45-AGF (#158)  
**Date:** 2026-08-09

---

## 1. Scope and design constraints

This document turns the v46 sparse-view generalization proposal (`docs/proposals/v46_sparse_view_generalization.md`) into a concrete, minimal implementation plan for the implementer agents. The design reuses existing infrastructure wherever possible:

- `VariableViewSetAggregator` / `PerceiverViewAggregator` for permutation-invariant view processing.
- `AdaptiveGeometryFusionV45` as an *optional* residual-based refinement (not a dependency).
- The existing `view_mask` plumbing in `OmniMultiViewFusionV5` and `experiments/train_omniview_fusion_v5_webbridge_multi.py`.
- The existing `augment_clip` view-dropout path for training-time augmentation.

No new heavy transformer or graph network is introduced. v46 is a **lightweight reliability head** that makes the existing v5/v25 triangulation path robust to dropped views.

---

## 2. `SparseViewGeneralizationV46` module API

### 2.1 File

`motionflow_mv/fusion/sparse_view_generalization_v46.py`

### 2.2 Class signature

```python
class SparseViewGeneralizationV46(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_views: int,
        hidden: int = 64,
        dropout: float = 0.0,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        num_inducing_points: int = 32,
    ) -> None:
        ...
```

| Argument | Meaning | Default |
|---|---|---|
| `in_channels` | Token dimension `d` (same as `OmniMultiViewFusionV5.d`). | required |
| `n_views` | Fixed view count expected by the model (for shape hints). | required |
| `hidden` | Hidden dimension of the per-view reliability MLP. | 64 |
| `dropout` | Dropout in the ISAB set aggregator. | 0.0 |
| `n_heads` | Attention heads in the ISAB. | 4 |
| `n_isab_layers` | Number of ISAB layers. | 2 |
| `num_inducing_points` | Inducing points per ISAB. | 32 |

### 2.3 Forward signature

```python
def forward(
    self,
    x: torch.Tensor,
    view_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        x: (B, T, V, J, C) multi-view feature tokens.
        view_mask: optional (B, T, V) or (B, V) bool/float mask.
                   1 / True means the view is available.

    Returns:
        reliability: (B, T, V, J) positive weights.
                     ~1.0 at initialization.
                     Zero for masked-out / missing views.
    """
```

### 2.4 Internal design

```text
Input tokens (B, T, V, J, C)
    |
    ▼
[ VariableViewSetAggregator ]   ← permutation-invariant ISAB; zeros masked views
    |
    ▼
Pool over joints: mean + std per (B, T, V)
    |
    ▼
[ reliability MLP ]               ← 2-layer MLP: 2*C -> hidden -> 1
    |
    ▼
2 * sigmoid(.)                  ← identity-like init (~1.0)
    |
    ▼
Broadcast to (B, T, V, J)
    |
    ▼
Multiply by view_mask
    |
    ▼
Output reliability (B, T, V, J)
```

Key invariants the module must guarantee:

1. **Shape:** output `(B, T, V, J)`.
2. **Positivity:** all weights `> 0` where unmasked.
3. **Identity at init:** weights in `(0.8, 1.2)` before masking so full-view training is not disturbed.
4. **Masking:** dropped views produce exactly zero weight.
5. **Gradient-friendly:** all parameters receive gradients under a standard backward pass.

### 2.5 Optional perceiver variant

If `use_perceiver_aggregator` is preferred over ISAB, a second implementation can be provided later that swaps `VariableViewSetAggregator` for `PerceiverViewAggregator`. For v46 MVP, the ISAB-backed module is sufficient because it reuses the same aggregator already present in `OmniMultiViewFusionV5`.

---

## 3. Integration into `OmniMultiViewFusionV5`

### 3.1 New constructor arguments

Add to `motionflow_mv/fusion/omniview_fusion_v5.py::OmniMultiViewFusionV5.__init__`:

```python
use_v46_sparse_view_generalization: bool = False,
v46_svg_hidden: int = 64,
v46_svg_dropout: float = 0.0,
v46_svg_n_heads: int = 4,
v46_svg_n_isab_layers: int = 2,
v46_svg_num_inducing_points: int = 32,
```

These are stored as instance attributes and used to instantiate the module only when the flag is on.

### 3.2 Module instantiation

```python
self.use_v46_sparse_view_generalization = use_v46_sparse_view_generalization
if self.use_v46_sparse_view_generalization:
    self.sparse_view_generalization_v46 = SparseViewGeneralizationV46(
        in_channels=self.d,
        n_views=n_views,
        hidden=v46_svg_hidden,
        dropout=v46_svg_dropout,
        n_heads=v46_svg_n_heads,
        n_isab_layers=v46_svg_n_isab_layers,
        num_inducing_points=v46_svg_num_inducing_points,
    )
else:
    self.sparse_view_generalization_v46 = None
```

### 3.3 Forward-path insertion

After the existing set/perceiver aggregator block (`omniview_fusion_v5.py` lines 1050–1054), call the v46 module:

```python
# Optional permutation-invariant view aggregator over views.
if self.use_perceiver_aggregator and self.perceiver_aggregator is not None:
    feat = self.perceiver_aggregator(feat, view_mask=view_mask)
elif self.use_set_view_aggregator and self.set_view_aggregator is not None:
    feat = self.set_view_aggregator(feat, view_mask=view_mask)

# >>> v46 sparse-view reliability head <<<
svg_weights = None
if self.use_v46_sparse_view_generalization and self.sparse_view_generalization_v46 is not None:
    svg_weights = self.sparse_view_generalization_v46(
        feat, view_mask=view_mask_flat.view(B, T, V)
    )  # (B, T, V, J)
```

`feat` has shape `(B, T, V, J, d)` at this point. The v46 head returns view reliability weights without altering the token shape.

### 3.4 Applying v46 weights in the main triangulation path

`confidences` is currently `(B * T, V, J)`. Reshape and multiply:

```python
# After the weight head produces per-view weights:
weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)

# Apply v46 sparse-view reliability if enabled.
if svg_weights is not None:
    weights = weights * svg_weights.view(B * T, V, J)

# Continue with the existing masking and precision terms.
weights = weights * confidences * precision * visibility
weights = weights.clamp(min=1e-4, max=1e4)
```

This is the minimal change: the v46 weights act as an additional multiplicative confidence factor before DLT triangulation.

### 3.5 Applying v46 weights in the v25 geometry-fusion path

When `use_multiview_geometry_fusion_v25=True`, the v25 block is called at `omniview_fusion_v5.py` lines 1427–1435. Pass the v46-weighted confidence:

```python
if self.use_multiview_geometry_fusion_v25 and self.multiview_geometry_fusion_v25 is not None:
    v25_confidence = confidences.view(B, T, V, J)
    if svg_weights is not None:
        v25_confidence = v25_confidence * svg_weights

    pred_3d_gn_v25, geom_loss_v25 = self.multiview_geometry_fusion_v25(
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        pred_3d_init=pred_3d_gn.view(B, T, J, 3),
        view_mask=view_mask_flat.view(B, T, V),
        confidence=v25_confidence,  # (B, T, V, J)
    )
    pred_3d_gn = pred_3d_gn_v25.view(B * T, J, 3)
```

Because `MultiViewGeometryFusionV25.forward` already multiplies the incoming `confidence` by `view_mask` before DLT, dropped views remain zero-weighted. If `use_v45_adaptive_geometry_fusion` is also on, v45 refines the *already v46-weighted* confidence, which is the desired order:

1. v46 coarse sparse-view reliability (dropout / mask based).
2. v45 fine residual-based reliability.
3. DLT / triangulation.

### 3.6 Interaction with `VariableViewSetAggregator` / `PerceiverViewAggregator`

v46 does **not** require the existing set aggregator to be enabled. It contains its own `VariableViewSetAggregator`, so it works in both configurations. If `use_set_view_aggregator=True` and `use_v46_sparse_view_generalization=True`, the view set is aggregated twice (once by the existing aggregator and once by v46). This is acceptable because both are lightweight and identity-like at init; it is not worth adding mutex logic for the MVP.

### 3.7 No changes to `MultiViewGeometryFusionV25` required

Because v46 weights are applied to `confidence` *before* calling v25, the v25 module can remain unchanged. This keeps the blast radius small and avoids conflicts with v45 and the outlier-view detector inside v25.

---

## 4. Training-loop view-dropout helper

### 4.1 File

`motionflow_mv/data/view_dropout_augmentation_v46.py`

### 4.2 Recommended API

```python
from typing import Tuple, Optional
import torch

def drop_views(
    x: torch.Tensor,
    prob: float = 0.3,
    min_views: int = 2,
    return_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Randomly drop full views during training.

    Args:
        x: (B, T, V, J, 3+) tensor of 2D keypoints + confidence + optional extras.
        prob: Probability of dropping each view.
        min_views: Minimum number of views to keep per sample.
        return_mask: If True, also return the boolean view mask.

    Returns:
        x: (B, T, V, J, 3+) tensor with dropped views' confidence zeroed.
        view_mask: (B, V) float mask (1 = kept, 0 = dropped) if return_mask=True.
    """
```

### 4.3 Implementation note

The helper is a thin wrapper around the existing `augment_clip` in `experiments/train_omniview_fusion_v5_webbridge_multi.py`. It should:

1. Draw a Bernoulli mask of shape `(B, V)` with keep probability `1 - prob`.
2. Enforce `min_views` by randomly re-activating dropped views if needed.
3. Zero out `x[..., 2]` (confidence) for dropped views.
4. Return `x` and the `(B, V)` mask.

The trainer can then broadcast the mask to `(B, T, V)` and pass it to the model as `view_mask`.

### 4.4 New CLI flags for the trainer

Add to `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

```python
parser.add_argument("--use_v46_sparse_view_generalization", action="store_true")
parser.add_argument("--v46_svg_view_dropout_prob", type=float, default=0.3)
parser.add_argument("--v46_svg_min_views", type=int, default=2)
parser.add_argument("--v46_svg_use_curriculum", action="store_true", default=True)
parser.add_argument("--v46_svg_hidden", type=int, default=64)
parser.add_argument("--v46_svg_dropout", type=float, default=0.0)
```

In the training loop, when `args.use_v46_sparse_view_generalization` is enabled:

```python
if args.use_v46_sparse_view_generalization:
    # Compose v46 dropout with existing augment_clip.
    x, v46_mask = drop_views(
        x,
        prob=curriculum_dropout_prob,  # see below
        min_views=args.v46_svg_min_views,
    )
    v46_mask = v46_mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, V)
    if view_mask is None:
        view_mask = v46_mask
    else:
        view_mask = view_mask * v46_mask
```

### 4.5 Curriculum

If `v46_svg_use_curriculum=True`, ramp the dropout probability from `0.0` at epoch 0 to `v46_svg_view_dropout_prob` at epoch `epochs // 2`, then hold:

```python
if args.v46_svg_use_curriculum:
    progress = min(1.0, epoch / max(1, args.epochs // 2))
    p = args.v46_svg_view_dropout_prob * progress
else:
    p = args.v46_svg_view_dropout_prob
```

This follows the existing `variable_view_training` curriculum style and avoids destabilizing early training.

---

## 5. Evaluation extension

### 5.1 File

`experiments/eval_variable_views.py`

### 5.2 New CLI flags

```python
parser.add_argument("--mpjpe_at_k", type=str, default="2,3,4,full",
                    help="Comma-separated list of view counts to report MPJPE@k for")
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--output_csv", type=str, default=None)
```

### 5.3 Reporting format

`evaluate_variable_views` should return a dict keyed by `k`:

```python
{
    2: {"mean_mm": 42.3, "std_mm": 1.2, "n_subsets": 6},
    3: {"mean_mm": 31.5, "std_mm": 0.8, "n_subsets": 4},
    4: {"mean_mm": 27.1, "std_mm": 0.5, "n_subsets": 1},
    "full": {"mean_mm": 26.9, "std_mm": 0.0, "n_subsets": 1},
}
```

CSV output columns:

```csv
k,mean_mm,std_mm,n_subsets
2,42.3,1.2,6
3,31.5,0.8,4
4,27.1,0.5,1
full,26.9,0.0,1
```

For `"full"`, evaluate with all `V` views active.

### 5.4 Integration with v46

When evaluating a v46-trained model, the existing `VariableViewInferenceWrapper` still works because the model obeys `view_mask`. The only change is that the v46 reliability head will additionally down-weight noisy/occluded views inside the model, improving few-view MPJPE.

---

## 6. Test requirements

Agent-12 should add `tests/test_sparse_view_generalization_v46.py` covering at minimum:

1. **Module unit tests**
   - Output shape `(B, T, V, J)` for `B=2, T=5, V=4, J=17, C=64`.
   - Positive weights.
   - Identity-like init: weights within `[0.8, 1.2]` for unmasked views.
   - Masked views produce exactly zero weight.
   - Gradient flow: `loss = out.sum(); loss.backward()` produces non-None gradients.

2. **Permutation equivariance**
   - Permuting input views permutes output weights identically (up to tolerance).

3. **Integration with `OmniMultiViewFusionV5`**
   - Model instantiates with `use_v46_sparse_view_generalization=True`.
   - Forward pass with random `view_mask` returns valid outputs.
   - Dropped views have near-zero final triangulation weights.

4. **Compatibility**
   - v46 + `use_v45_adaptive_geometry_fusion=True` runs without error.
   - v46 + `use_multiview_geometry_fusion_v25=True` runs without error.
   - v46 + `use_set_view_aggregator=True` runs without error.

5. **View-dropout helper**
   - `drop_views` returns a mask of shape `(B, V)`.
   - Enforces `min_views`.
   - Zeros confidence of dropped views.

---

## 7. Smoke config and launch script

### 7.1 Smoke config

`configs/benchmark_v46_svg_smoke.yaml` should enable:

```yaml
use_v46_sparse_view_generalization: true
v46_svg_view_dropout_prob: 0.3
v46_svg_min_views: 2
v46_svg_use_curriculum: true
v46_svg_hidden: 64
v46_svg_dropout: 0.0
```

Keep the rest of the config identical to the v25/v45 smoke baseline so the only variable is v46.

### 7.2 Local smoke script

`scripts/run_v46_svg_smoke_local_4090.sh` should run the trainer in smoke mode with the new flags:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --config configs/benchmark_v46_svg_smoke.yaml \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2
```

### 7.3 A800 queue entry

Agent-14 should append an entry to `scripts/launch_v33_a800_queue.py`:

```python
{
    "variant": "v46_svg",
    "config": "configs/benchmark_v46_svg_smoke.yaml",
    "gpu_mem_required_gb": 30,
    # ... other queue metadata
}
```

The full-run config should inherit from the smoke config but use the full WebBridge all-train manifest and longer training schedule.

---

## 8. Compatibility and no-overdesign checklist

| Concern | Decision |
|---|---|
| Reuse existing variable-view code | Yes — `view_mask` plumbing, `augment_clip`, and `VariableViewSetAggregator` are reused. |
| Depend on v45-AGF | No — v46 is self-contained; v45 can be enabled on top. |
| Modify `MultiViewGeometryFusionV25` | No — v46 weights are applied to `confidence` before the call. |
| Add new heavy architecture | No — only ISAB + 2-layer MLP. |
| Change dataset/loader | No — dropout is applied on-the-fly in the training loop. |
| Break full-view baseline | No — identity-like init keeps behavior unchanged at start. |

---

## 9. Open questions / potential blockers

1. **GPU availability:** The local RTX 4090 is running the v45-AGF medium run. v46 smoke must wait until that GPU frees (per AGENTS.md).
2. **Interaction with `use_domain_embedding`:** When training on mixed H36M/MPI/WebBridge data, the domain-specific base `view_mask` should be composed with the v46 dropout mask by multiplication, as the trainer already does. This should be verified in the integration test.
3. **Per-view vs. per-joint weights:** The current design outputs per-view reliability broadcast to `(B, T, V, J)`. If later experiments show per-joint weights help, the MLP can be extended without changing the integration API.
4. **Minimum views for triangulation:** The v46 helper must enforce `min_views >= 2`. With exactly 2 active views, DLT is well-defined but noisier; the v46 reliability head still helps by down-weighting the worse of the two views.

---

## 10. Summary for implementers

1. **Agent-06** — `motionflow_mv/fusion/sparse_view_generalization_v46.py`: implement `SparseViewGeneralizationV46` with the API above. Ensure identity init, masking, and gradient tests pass.
2. **Agent-08** — `motionflow_mv/fusion/omniview_fusion_v5.py`: add the v46 flags, instantiate the module, and wire it into the forward path as described in Sections 3.3–3.5.
3. **Agent-09** — `experiments/train_omniview_fusion_v5_webbridge_multi.py`: add CLI flags and call the view-dropout helper before the existing `view_mask` composition.
4. **Agent-07** — `motionflow_mv/data/view_dropout_augmentation_v46.py`: implement the thin `drop_views` wrapper.
5. **Agent-13** — `experiments/eval_variable_views.py`: extend to emit `MPJPE@k` CSV/JSON for `k = 2,3,4,full`.
6. **Agent-12** — `tests/test_sparse_view_generalization_v46.py`: write unit/integration tests per Section 6.
7. **Agent-10/11** — create smoke config and launch script.
8. **Agent-14** — add the A800 queue entry.

This design keeps v46 as a small, isolated reliability head that sits on top of the existing v5/v25 pipeline, generalizing the model to sparse and variable camera rigs without overhauling the architecture.
