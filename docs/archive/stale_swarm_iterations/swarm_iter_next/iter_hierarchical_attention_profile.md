# Hierarchical Attention Model CPU Profile

**Date:** 2026-08-06  
**Commit:** `1fcbaa3e3b525ff186492f8957c931c2e18c59e5`  
**Environment:** Windows, PyTorch 2.13.0+cpu, CPU-only (`CUDA_VISIBLE_DEVICES=-1`)  
**Script:** `tmp/profile_hierarchical_speed.py`

## Goal

Profile the `hierarchical_view_temporal_joint_pp` model to understand why it is CPU-bound and slow, using a tiny synthetic batch and small dimensions without touching the GPU.

## Methodology

The profile script:

- Forces CPU execution via `CUDA_VISIBLE_DEVICES=-1`.
- Builds a synthetic batch of `B=2, T=13, V=14, J=28` with random 2D keypoints/confidences and pinhole cameras.
- Instantiates the model with small dimensions:
  - `d=32`, `residual_hidden=64`
  - `n_st_layers=1`, `n_view_layers=1`, `n_temporal_layers=1`, `n_joint_graph_layers=1`
  - `n_view_groups=2`, `n_heads=4`
- Runs 2 warmup iterations and 5 measured forward+backward passes.
- Instruments key submodules with forward hooks and monkey-patches `_triangulate_weighted_dlt` for timing.
- Captures a PyTorch autograd profiler trace for per-operator breakdown.

## Results

### Per-epoch wall-clock time

| Metric | Value |
|---|---|
| Avg. forward+backward (CPU) | **0.564 s** |
| Projected epochs/min | ~106 |

> Note: this is with the deliberately tiny configuration requested in the task. Full dimensions and more layers will scale these numbers up, but the relative ranking of expensive stages remains informative.

### Forward stage timing (CPU seconds per pass)

| Rank | Stage / Module | Avg. Time | % of slowest |
|---|---|---|---|
| 1 | `_triangulate_weighted_dlt` | 0.0947 s | 100.0 % |
| 2 | `hierarchical_block` (total) | 0.0886 s | 93.5 % |
| 3 | `joint_attn.0` (base per-frame encoder) | 0.0492 s | 51.9 % |
| 4 | `hierarchical_block.temporal_layers.0` | 0.0392 s | 41.4 % |
| 5 | `hierarchical_block.view_layers.0` | 0.0198 s | 20.9 % |
| 6 | `hierarchical_block.joint_graph_layers.0` | 0.0052 s | 5.5 % |
| 7 | `principal_point_correction` | 0.0021 s | 2.2 % |
| 8 | `weight_head` | 0.0003 s | 0.3 % |
| 9 | `residual_mlp` | 0.0007 s | 0.8 % |

Inside the `hierarchical_block`, the **temporal transformer layer** is the most expensive sub-component, followed by the **within-group view attention**. The skeleton-graph (`GraphJointRelation`) layer is comparatively cheap at this size.

### PyTorch profiler top CPU operators

| Name | Self CPU % | Note |
|---|---|---|
| `aten::dropout` / `aten::bernoulli_` | ~21.9 % | Random mask generation in every TransformerEncoderLayer |
| `aten::scaled_dot_product_attention` | ~9.6 % | Attention math (forward + backward) |
| `aten::linalg_lstsq` | ~5.5 % | DLT triangulation per joint |
| `aten::mul`, `aten::copy_`, `aten::select` | various | Autograd/reshape/data-movement overhead |
| `LinalgLstsqBackward0` | ~6.4 % | Backward through `linalg_lstsq` |

## Bottleneck Interpretation

1. **DLT triangulation (`_triangulate_weighted_dlt`) is the single most expensive stage.**
   - The implementation loops over joints and calls `torch.linalg.lstsq` once per joint.
   - For `B=2, T=13, J=28`, that is already **728 separate `lstsq` calls** per forward pass.
   - `lstsq` is also expensive in backward mode (`LinalgLstsqBackward0` shows up separately).

2. **Transformer attention layers dominate the `hierarchical_block`.**
   - The temporal layer is the most expensive, then the view layer.
   - Dropout (`aten::dropout` / `aten::bernoulli_`) accounts for a large share of CPU time because every TransformerEncoderLayer samples a dropout mask.
   - These layers also involve many `permute`/`reshape`/`view` operations that generate autograd overhead (`SliceBackward0`, `SelectBackward0`).

3. **Skeleton graph (`GraphJointRelation`) is *not* the bottleneck at this scale.**
   - It is only ~5.5 % of the slowest stage.
   - However, its current implementation (`index_add_` + Python loops over edge types) can still be a CPU/GPU sync point and may become heavier with larger `V` or `J`.

4. **CPU-bound behavior on an RTX 4090 run likely stems from:**
   - The per-joint `linalg.lstsq` loop creating many small CPU/GPU synchronization points.
   - Python-level loops in `_triangulate_weighted_dlt` (over joints) and in `GraphJointRelation` (over edge types).
   - Frequent reshape/permute/copy operations that keep the CPU busy feeding the GPU.
   - Dropout randomness and autograd bookkeeping overhead on CPU.

## Suggested Fixes

1. **Batch the DLT triangulation.**
   - Replace the per-joint `for j in range(J): ... torch.linalg.lstsq(...)` loop with a batched `torch.linalg.lstsq(Aw, bw)` where `Aw` and `bw` have shape `(B*T*J, 2V, 3)` and `(B*T*J, 2V, 1)`.
   - This turns hundreds of small solves into one batched kernel call, dramatically reducing kernel-launch and CPU-sync overhead on GPU.

2. **Reduce transformer overhead.**
   - Ensure training runs on GPU; the CPU profile confirms transformer ops are expensive when forced onto CPU.
   - If attention remains a bottleneck on GPU, consider fused kernels (e.g. FlashAttention-compatible implementations) or reduce `dim_feedforward`.
   - Minimize unnecessary `permute`/`reshape` chains in the hierarchical block; fuse where possible.

3. **Optimize `GraphJointRelation` if scaled up.**
   - Vectorize the edge-type projection: apply one `Linear(d, d)` per edge type to the whole edge source tensor, then use `torch.where` or `scatter` with precomputed edge-type masks instead of the Python `for t in range(3):` loop.
   - Pre-compute and register edge-type masks as buffers.
   - Use `torch.scatter_add` or `torch.sparse` for aggregation.

4. **Reduce dropout / data-movement overhead.**
   - Use `nn.Dropout` in-place variants or fused dropout where supported.
   - Avoid repeated `index_add_` and `permute` chains that keep tensors in non-contiguous layouts.

## Blockers / Caveats

- This profile is **CPU-only**. GPU timing will differ, but the identified hotspots (triangulation loop and attention/dropout) are the same operations that would stall a GPU run.
- No source code was modified. The monkey-patching is confined to the profile script.
- The tiny configuration keeps absolute times small; full-scale bottlenecks may shift if, for example, the graph layer grows faster with `V` or `J` than the transformer layers.
