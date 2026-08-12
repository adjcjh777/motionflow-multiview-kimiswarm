# Real-Time Efficiency Benchmark Report

**Date**: 2026-08-12 12:49:34 UTC  
**Device**: `cuda`  
**Input shape**: (B, T=13, V=14, J=28, 3)  
**Warmup iterations**: 20  
**Measured iterations**: 100  

## Model Variants

| Model | Params | Single-frame (ms) | Single FPS | Clip (ms) | Throughput (fps) | Peak Mem (MB) | 60 FPS? | 30 FPS? |
|-------|--------:|------------------:|-----------:|----------:|-----------------:|----------------:|--------:|--------:|
| RayAttentionFusionModelV3 | 134,497 | 86.98 | 11.5 | 86.98 | 45.0 | 10.6484375 | No | No |
| RayAttentionFusionModelTemporal | 217,825 | 88.53 | 11.3 | 86.61 | 575.2 | 24.9033203125 | No | No |
| RayAttentionFusionModelTemporalResidual | 243,428 | 87.20 | 11.5 | 83.57 | 596.2 | 25.00146484375 | No | No |

## Interpretation

- **Single-frame latency**: time to process one `(B=1, T=1)` multi-view frame. This is the most relevant metric for streaming/real-time deployment.
- **Clip latency**: time to process one `(B=1, T=13)` clip, typical of the temporal model's inference unit.
- **Batch throughput**: frames per second when processing `(B=4, T=13)` batches; represents offline / batched throughput rather than streaming latency.
- **Peak memory**: maximum GPU memory allocated during a single `(B=1, T=13)` forward pass; CPU runs report `N/A`.

## Real-time feasibility

A model is marked as meeting a real-time target when its single-frame latency leaves at least 50 % of the frame budget free for preprocessing, I/O, and downstream pipeline stages. For 60 Hz streaming the budget is `1000 / 60 * 0.5 ~ 8.33 ms`; for 30 Hz it is `1000 / 30 * 0.5 ~ 16.67 ms`.

## Notes

- This benchmark uses randomly initialized weights; reported numbers reflect architecture-level latency/throughput and are independent of learned weights.
- Synthetic inputs and a fixed camera rig remove any dataset dependency, so the script can be run on any CUDA or CPU host as a quick smoke test.
- The `RayAttentionFusionModelTemporalResidual` is the current best model and therefore the primary target for RTX 4090 real-time evaluation.

## Methodology and design decisions

1. **Synthetic fixed rig** - Reuses the pure-torch `_FixedRig` helper from `experiments/benchmark_residual_temporal.py` so the benchmark is dataset-agnostic and runs on any host.
2. **Single-frame latency is the real-time gate** - Streaming applications must process each incoming frame before the next one arrives, so `B=1, T=1` latency is reported as the primary real-time metric.
3. **Clip and batch metrics for throughput** - Clip latency shows the cost of temporal models that operate on `(B=1, T=13)` windows, while `B=4` throughput estimates offline/batched capacity.
4. **Memory headroom** - GPU peak memory is captured via `torch.cuda.max_memory_allocated` to flag models that may exhaust frame-buffer budgets on edge devices.
5. **Real-time target with 50 % slack** - A model is considered feasible for a given frame rate only if it uses at most half of the frame budget, leaving room for preprocessing, I/O, and downstream stages.

## Limitations and future work

- CPU numbers here are for smoke-test verification only; the intended target hardware is an NVIDIA RTX 4090.
- `torch.compile` and TensorRT/ONNX export are not evaluated in this baseline; both can materially improve latency and should be benchmarked in follow-up work.
- The per-joint count defaults to 28 and view count to 14, matching the MPI-INF-3DHP setup used by the current best model.