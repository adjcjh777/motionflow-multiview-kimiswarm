# v49: Real-Time Streaming Multi-View Pose Estimation

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #165 (proposed)  
**Depends on:** v47-temporal (#162), v48-domain (#164)  

## Motivation

v46 Sparse-View Generalization, v47 Temporal Aggregation, and v48 Domain Generalization have pushed accuracy and robustness on studio and in-the-wild data. The next bottleneck for practical deployment is **real-time, streaming inference**: the current stack is still clip-based, uses full-window temporal self-attention, and keeps every view in the forward pass. For live camera rigs, sports analysis, and AR/VR pipelines we need a model that can run at **30–60 Hz** with **bounded latency and memory** while preserving the accuracy of the v48 base.

v49 therefore focuses on three real-time goals:

1. **Causal streaming.** Process each new frame as it arrives, without access to future frames, and keep a small, fixed-size state instead of an ever-growing clip buffer.
2. **Dynamic view budget.** Use the v46 reliability head to spend compute only on the most informative views, dropping redundant cameras on easy frames.
3. **Bounded memory and latency.** Keep per-frame cost O(1) in clip length and offer a clear accuracy/latency trade-off through a small set of knobs.

## Design principles

1. **Build on v48.** v49 is an optional *mode* that reuses v46 reliability, v47 temporal-window infrastructure, and v48 domain-invariant features. It does not replace the heavy v48 stack for offline use.
2. **Streaming-first.** Every new component must be causal and stateful; clip-based training is supported through causal frame-by-frame processing.
3. **Identity at initialization.** New neural heads start as no-ops, so v49 can be warm-started from a v48 checkpoint without regressing accuracy.
4. **Minimal new modules.** Only one new neural head (`StreamingTemporalSmootherV49`) and one inference-time policy (`DynamicViewBudgetV49`) are required.

## Proposed architecture

```text
Input: 2D keypoints + cameras for frame t
        |
        v
[v25 Multi-View Geometry Fusion]
        |
        v
[v46 Sparse-View Generalization reliability weights r_t]
        |
        |---> DynamicViewBudgetV49 (optional)
        |       mask out low-reliability views before heavy heads
        |
        v
Per-frame triangulated pose P_t  (B, J, 3)
        |
        v
[ v49 Streaming Temporal Smoother ]
        |
        ├── Causal GRU/MLP over a short rolling window
        ├── View-count confidence gating
        └── Output refined pose P'_t  (B, J, 3)
```

### Module 1: `StreamingTemporalSmootherV49`

**File:** `motionflow_mv/fusion/realtime_streaming_v49.py`

A lightweight, causal alternative to the full-clip `TemporalAggregationV47`. It processes one frame at a time and maintains a compact hidden state.

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple


class StreamingTemporalSmootherV49(nn.Module):
    """Causal, stateful temporal smoother for real-time streaming.

    Parameters
    ----------
    n_joints:
        Number of skeleton joints ``J``.
    d_model:
        Hidden dimension of the internal GRU/MLP.
    n_heads:
        Unused when ``use_gru=True``; kept for API symmetry with v47.
    num_layers:
        Number of MLP layers after the GRU hidden state.
    window:
        Number of past frames kept in a rolling buffer when ``use_gru=False``.
        With the default GRU path, ``window`` only affects the training mask.
    use_gru:
        If ``True`` use a single GRU cell for the state update (cheap, O(1)).
        If ``False`` use a causal 1-D conv over the rolling window.
    dropout:
        Dropout applied to the MLP layers.
    """

    def __init__(
        self,
        n_joints: int = 17,
        d_model: int = 32,
        n_heads: int = 4,
        num_layers: int = 1,
        window: int = 5,
        use_gru: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.d_model = d_model
        self.use_gru = use_gru
        self.window = window

        in_dim = n_joints * 3

        self.input_proj = nn.Linear(in_dim, d_model)

        if use_gru:
            self.gru = nn.GRUCell(d_model, d_model)
        else:
            self.register_buffer("_window_buffer", torch.zeros(1, window, n_joints * 3))
            self.temporal_conv = nn.Conv1d(
                in_channels=n_joints * 3,
                out_channels=d_model,
                kernel_size=window,
                padding=0,
                bias=True,
            )

        mlp = []
        for _ in range(num_layers):
            mlp.extend([nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*mlp)

        self.output_proj = nn.Linear(d_model, n_joints * 3)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Soft residual gate; identity behaviour at init.
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        pose_t: torch.Tensor,                       # (B, J, 3)
        hidden: Optional[torch.Tensor] = None,      # (B, d_model)
        view_count_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (refined_pose_t, updated_hidden)."""
        B, J, _ = pose_t.shape
        x = pose_t.reshape(B, -1)                  # (B, J*3)
        x = self.input_proj(x)                     # (B, d_model)

        if hidden is None:
            hidden = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)

        if self.use_gru:
            h = self.gru(x, hidden)                # (B, d_model)
        else:
            h = hidden + x                           # simple additive update for conv path

        feat = self.mlp(h)
        delta = self.output_proj(feat).reshape(B, J, 3)

        # Optionally discount the residual when few views are available.
        if view_count_t is not None:
            scale = (view_count_t.float() / (view_count_t.float() + 1.0)).view(B, 1, 1)
            delta = delta * scale

        refined = pose_t + torch.sigmoid(self.residual_gate) * delta
        return refined, h
```

**Key properties**

- **Causal and O(1).** With the GRU path, each frame costs one GRU cell update plus two small linear layers; cost does not grow with clip length.
- **Identity at init.** `output_proj` and `residual_gate` are initialized to zero, so the module is a no-op until trained.
- **Variable views.** `view_count_t` scales the residual so frames with fewer views are smoothed less aggressively.

### Module 2: `DynamicViewBudgetV49`

A small inference-time policy (no learnable parameters) that prunes views before the heavy geometry and temporal heads.

```python
class DynamicViewBudgetV49:
    """Hard-prune the least reliable views based on v46 reliability weights.

    Parameters
    ----------
    max_views:
        Maximum number of views to keep per frame. ``None`` disables pruning.
    min_views:
        Minimum number of views that must survive pruning.
    """

    def __init__(self, max_views: Optional[int] = None, min_views: int = 2):
        self.max_views = max_views
        self.min_views = min_views

    def __call__(self, reliability: torch.Tensor) -> torch.Tensor:
        """Return a binary view mask of the same shape as ``reliability``.

        Args
        ----
        reliability:
            ``(B, V, J)`` positive weights from ``SparseViewGeneralizationV46``.
        """
        if self.max_views is None or self.max_views <= 0:
            return torch.ones_like(reliability[..., 0])

        B, V, J = reliability.shape
        # Per-view score: average over joints.
        score = reliability.mean(dim=-1)            # (B, V)
        k = min(self.max_views, V)
        k = max(k, self.min_views)
        _, topk = torch.topk(score, k, dim=-1)      # (B, k)

        mask = torch.zeros(B, V, device=reliability.device, dtype=torch.bool)
        mask.scatter_(1, topk, True)
        return mask
```

The returned mask is applied to `view_mask` before the v25 geometry fusion block, reducing triangulation and attention cost roughly in proportion to `max_views / V`.

### Optional: confidence-conditioned fast path

A simple latency scheduler can skip the temporal smoother on high-confidence frames:

```python
confidence = view_count_t * reliability.mean()
if confidence > v49_confident_threshold:
    pose_t = per_frame_pose_t          # fast path: skip STS
else:
    pose_t, hidden = smoother(pose_t, hidden, view_count_t)
```

This is disabled by default (`v49_confident_threshold = inf`) and can be enabled once the reliability head is calibrated.

## Integration plan

### Files touched (future IMPLEMENT/EVAL tasks)

- `motionflow_mv/fusion/realtime_streaming_v49.py` — new `StreamingTemporalSmootherV49` and `DynamicViewBudgetV49`.
- `motionflow_mv/fusion/omniview_fusion_v5.py` — v49 flags, instantiate the streaming smoother, and branch the forward pass.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — CLI flags, streaming-mode training loop, optional distillation loss.
- `experiments/eval_variable_views.py` — streaming MPJPE@k and latency/memory reporting.
- `configs/benchmark_v49_streaming_smoke.yaml` — smoke config.
- `scripts/run_v49_streaming_smoke_local_4090.sh` — smoke script.

### New training/evaluation flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_v49_streaming` | `False` | Master switch for real-time streaming mode. |
| `v49_streaming_d_model` | `32` | Hidden size of the streaming GRU/MLP. |
| `v49_streaming_num_layers` | `1` | Number of post-GRU MLP layers. |
| `v49_streaming_use_gru` | `True` | Use GRU (`True`) vs causal 1-D conv (`False`). |
| `v49_streaming_window` | `5` | Rolling window size for the conv path and causal training mask. |
| `v49_view_budget` | `None` | If set, hard-limit the number of active views per frame. |
| `v49_min_views` | `2` | Minimum surviving views after pruning. |
| `v49_confident_threshold` | `1e9` | Confidence threshold for the optional fast path (disabled by default). |

### Constructor additions in `OmniMultiViewFusionV5`

```python
# Optional v49 real-time streaming head.
self.use_v49_streaming = use_v49_streaming
if self.use_v49_streaming:
    from motionflow_mv.fusion.realtime_streaming_v49 import (
        StreamingTemporalSmootherV49,
        DynamicViewBudgetV49,
    )

    self.streaming_smoother = StreamingTemporalSmootherV49(
        n_joints=self.j,
        d_model=v49_streaming_d_model,
        n_heads=self.n_heads,
        num_layers=v49_streaming_num_layers,
        window=v49_streaming_window,
        use_gru=v49_streaming_use_gru,
        dropout=v49_streaming_dropout,
    )
    self.dynamic_view_budget = DynamicViewBudgetV49(
        max_views=v49_view_budget,
        min_views=v49_min_views,
    )
```

### Forward path

The v49 branch is inserted **after** the v46 triangulation has produced per-frame 3D poses:

```python
# Inside OmniMultiViewFusionV5.forward(...)
poses_3d = triangulate(...)  # (B, T, J, 3)

if self.use_v49_streaming and streaming:
    # Optional dynamic view budget already applied to view_mask earlier.
    B, T, J, _ = poses_3d.shape
    hidden = None
    outputs = []
    for t in range(T):
        view_count_t = view_mask[:, t].sum(dim=-1)  # (B,)
        refined_t, hidden = self.streaming_smoother(
            poses_3d[:, t], hidden, view_count_t=view_count_t
        )
        outputs.append(refined_t)
    poses_3d = torch.stack(outputs, dim=1)
else:
    # Fall back to the clip-based v47 head when available.
    if self.use_v47_temporal_aggregation:
        poses_3d = self.temporal_aggregation_v47(poses_3d, view_mask=view_mask)
```

A cleaner production API is a separate method `streaming_forward_frame(...)` that accepts a single frame and returns the refined pose plus the updated hidden state, so the host application can call it from a real-time callback.

## Training recipe

1. **Warm-start from the best v48 checkpoint.** Freeze v25/v45/v46/v48 weights for the first epoch and train only the new `StreamingTemporalSmootherV49` head.
2. **Causal clip training.** During training, feed clips of length `T` but reset the hidden state at sequence boundaries and apply a causal mask so the smoother never sees future frames.
3. **Unfreeze and fine-tune.** After the warm-start, unfreeze the backbone and fine-tune end-to-end for a few epochs with the v48 mixed manifest plus 3DPW actual.
4. **Optional distillation.** If accuracy drops more than 0.5 mm, add a distillation loss that matches the v48 batch-mode output on the same frames.

## Evaluation

Extend `experiments/eval_variable_views.py` (or create `experiments/eval_streaming_v49.py`) to report:

| Metric | Description |
|--------|-------------|
| `MPJPE@k` | Standard pose error at view count `k`. |
| `streaming_MPJPE@k` | Pose error when the model processes the clip causally frame-by-frame. |
| `latency_ms` | Per-frame forward latency on RTX 4090 (`B=1, T=1`). |
| `clip_latency_ms` | Latency for a short causal window (`B=1, T=5`). |
| `peak_mem_MB` | Peak GPU memory during streaming inference. |
| `throughput_fps` | Frames per second in a streaming loop. |

Suggested real-time targets on RTX 4090:

- **30 Hz streaming:** per-frame latency ≤ 16.67 ms, MPJPE within 0.5 mm of v48 batch.
- **60 Hz streaming:** per-frame latency ≤ 8.33 ms, MPJPE within 1.0 mm of v48 batch.

## Success criteria

1. Smoke test passes with no NaN/OOM and finite MPJPE on H36M/MPI/AIST/3DPW actual.
2. v49 streaming MPJPE at full views is within 0.5 mm of the v48 batch baseline on the same validation split.
3. Per-frame streaming latency on RTX 4090 is ≤ 16.67 ms; target ≤ 8.33 ms.
4. Dynamic view budget with `v49_view_budget=2` yields MPJPE within 1.5× of the full-view v48 baseline.
5. Peak memory for `B=1, T=1` is ≤ 1 GB.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Causal temporal head lags behind the non-causal v47 head. | Warm-start from v48; use distillation if needed. |
| View pruning drops informative cameras. | Enforce `min_views=2`; use v46 reliability, not random pruning. |
| Hidden-state management across clip boundaries is error-prone. | Expose explicit `reset_state()` and `streaming_forward()` methods; reset at sequence boundaries in the trainer. |
| Dynamic shapes break `torch.compile`/ONNX export. | Fix `window`, `J`, and `V` in deployment configs; export only the neural head and keep triangulation as a post-process. |
| Accuracy-latency trade-off is too steep. | Provide three preset modes: `quality`, `balanced`, `fast`. |

## Relation to other variants

- **v46 Sparse-View Generalization:** v49 reuses the v46 reliability weights for dynamic view budgeting; v46 remains the base.
- **v47 Temporal Aggregation:** v49 provides a causal, streaming alternative to the full-clip v47 transformer. They can coexist: v47 for batch, v49 for streaming.
- **v48 Domain Generalization:** v49 operates on domain-invariant features produced by v48; no domain-specific logic is needed.
- **v30 Adaptive Online Self-Evolution:** v30’s adaptive compute idea is re-used in the confidence-conditioned fast path, but v49 targets real-time latency rather than TTE depth.
- **P19 Real-Time Plan:** v49 directly addresses the streaming-temporal-window and early-view-pruning items in the P19 roadmap.

## Next steps

1. Wait for v47-temporal smoke (#162) and v48-domain smoke (#164) to land.
2. Implement `StreamingTemporalSmootherV49` and `DynamicViewBudgetV49`.
3. Wire the v49 flags into `OmniMultiViewFusionV5` and the trainer.
4. Add `configs/benchmark_v49_streaming_smoke.yaml` and `scripts/run_v49_streaming_smoke_local_4090.sh`.
5. Run smoke on RTX 4090 and measure the accuracy/latency Pareto frontier versus v48.
6. Queue a full A800 run only after the local smoke meets the latency targets.
