# Iter11+ Real-Time Efficiency Roadmap

## 1. Current state and bottlenecks

The new combined model `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` (`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`) fuses ray-aware per-view embeddings, spatio-temporal cross-view attention, uncertainty-weighted DLT, differentiable Gauss–Newton refinement, and a residual MLP. It reaches ~11.17 mm MPJPE on MPI-INF-3DHP but has a large compute footprint. The dominant bottlenecks are:

1. **Spatio-temporal attention.** The model reshapes `(B, T, V, J, d)` into `(B*J, T*V, d)` and runs full self-attention. Cost per layer is `O((T·V)²·d)` — about **1.5 M** token pairs per joint for `T=13, V=14`.
2. **Per-frame encoder.** View-level attention over `V` tokens and joint-level `TransformerEncoderLayer` over `J` tokens are repeated for every frame `T`.
3. **Refinement stack.** A -iteration Gauss–Newton solver and a residual loop sit on top of the DLT triangulation.

Existing benchmarks (`experiments/benchmark_inference_v3.py`, `experiments/benchmark_residual_temporal.py`) measure latency and throughput, but they do not profile the combined model, report FLOPs, or test CPU real-time feasibility.

## 2. Proposed improvements

### 2.1 Factorized spatio-temporal attention

Replace the full `(time × view)` self-attention with a separable scheme:

- temporal attention over `T` tokens per `(view, joint)` pair;
- cross-view attention over `V` tokens per `(time, joint)` pair.

Cost drops from `O(T²V²)` to `O(T²V + TV²)` — a **~7×** reduction for `T=13, V=14`.

### 2.2 FlashAttention / SDPA

`nn.MultiheadAttention` uses the legacy attention path. Switching to `torch.nn.functional.scaled_dot_product_attention` enables FlashAttention on Ampere/Ada GPUs, cutting attention latency and memory by ~2×.

### 2.3 Slimmer architecture

- Reduce `d` from `64` to `32` or `48` in the spatio-temporal block; keep the per-frame encoder at `d=64` and project.
- Reduce `n_st_layers` from `2` to `1` and `n_joint_layers` to `0` or `1`.
- Shrink the residual MLP hidden size from `128` to `64`.

Together these can cut FLOPs by **30–50%**. Accuracy can be preserved via distillation.

### 2.4 Local / streaming temporal attention

Use a fixed local temporal window (`w=5`) instead of global attention, or implement a causal streaming cache that updates features frame-by-frame. This makes per-frame cost `O(1)` rather than `O(T)`.

### 2.5 Cheaper refinement

- Make `gn_iters` a runtime parameter and default to `1` (or `0`) at inference.
- Fix `n_iter` in the residual loop to `1` for inference.
- Vectorize `_triangulate_weighted_dlt` over joints: build a batched system of shape `(B, J, 2V, 4)` and call `torch.linalg.lstsq` once instead of looping.

### 2.6 Knowledge distillation

Train a small student (`d=32`, one factorized st layer, no GN) to match the full teacher. Use L2 loss on 3D outputs and optional token-alignment loss. This keeps most accuracy while giving a tiny deployable model.

### 2.7 Deployment optimizations

- Run the network under `torch.autocast(..., dtype=torch.float16)` while keeping triangulation in FP32.
- Apply `torch.compile(..., mode="reduce-overhead")` for CUDA graphs.
- Export only the neural part to ONNX/TensorRT; run triangulation in a separate geometry step. Full-model ONNX export already fails because `torch.linalg.lstsq` is unsupported.

## 3. Recommended experiments

1. **Profile the combined model.** Add `benchmark_inference_combined.py` reporting per-component latency and memory for `V ∈ {4, 14}` and `T ∈ {1, 9, 13, 27}`.
2. **Factorized vs. full attention ablation.** Train both variants on the same data and compare MPJPE and latency.
3. **Dimension/depth sweep.** Grid over `d ∈ {32, 48, 64}`, `n_st_layers ∈ {1, 2}`, `gn_iters ∈ {0, 1, 3}` and report the MPJPE vs. latency Pareto frontier.
4. **Distillation run.** Train the `d=32, n_st_layers=1` student from the `d=64, n_st_layers=2` teacher.
5. **Deployment run.** Export the chosen head to ONNX/TensorRT and measure final B=1 latency.

## 4. Metrics to track

| Metric | Purpose |
|--------|---------|
| Latency (ms), B=1, CPU & GPU | Real-time feasibility |
| Throughput (fps) | Batch headroom |
| FLOPs / params | Architectural efficiency |
| MPJPE / PA-MPJPE | Accuracy |
| PCK@50/100/150 mm, AUC | Robustness |
| Peak memory (MB) | Deployment envelope |

Target: **< 33 ms per clip on GPU (≥30 FPS) and < 100 ms on CPU, with MPJPE within 0.5 mm of the current 11.17 mm baseline**.

## 5. Risks

- **Factorization weakens cross-view/temporal interactions.** Mitigate with a final lightweight cross-attention or keep one full layer.
- **FlashAttention unavailable on CPU/older GPUs.** SDPA falls back to the math path automatically.
- **Smaller `d`/fewer layers hurt accuracy.** Use distillation and tune via the Pareto sweep.
- **ONNX export blocked by `lstsq`.** Keep triangulation as a post-process and export only the neural head.
- **CPU still too slow.** Use local windows or frame skipping for CPU targets.

## 6. Code sketch: factorized spatio-temporal layer

```python
import torch.nn as nn

class FactorizedSTLayer(nn.Module):
    def __init__(self, d, n_heads, n_views):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d)
        self.view_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.view_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d)
        )
        self.ffn_norm = nn.LayerNorm(d)

    def forward(self, x):
        # x: (B, T, V, J, d)
        B, T, V, J, d = x.shape

        # Temporal attention: (B*V*J, T, d)
        x_t = x.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, d)
        out, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = self.temporal_norm(x_t + out)
        x = x_t.view(B, V, J, T, d).permute(0, 3, 1, 2, 4)

        # Cross-view attention: (B*T*J, V, d)
        x_v = x.permute(0, 2, 3, 1, 4).reshape(B * T * J, V, d)
        out, _ = self.view_attn(x_v, x_v, x_v)
        x_v = self.view_norm(x_v + out)
        x = x_v.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)

        x = self.ffn_norm(x.reshape(B * T * V * J, d) + self.ffn(x.reshape(B * T * V * J, d)))
        return x.view(B, T, V, J, d)
```

This block directly replaces the `st_transformer` in the combined model without changing the rest of the pipeline.

## 7. Next steps

1. Land the profiling benchmark and collect combined-model baseline latency/FLOPs.
2. Implement and train the factorized `STLayer` ablation.
3. Run the dimension/depth sweep and pick the configuration meeting the 33 ms / 0.5 mm target.
4. Export the chosen model head to ONNX/TensorRT and measure real-time latency.
