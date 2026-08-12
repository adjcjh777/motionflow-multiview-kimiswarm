# Runtime & Efficiency Benchmark on RTX 4090

## 1. Current state

A dedicated inference benchmark already exists for the best architecture, `RayAttentionFusionModelTemporalResidual`, and is documented in the paper draft.

**Relevant files**

- `experiments/benchmark_residual_temporal.py:1–252` — end-to-end latency/throughput benchmark for `RayAttentionFusionModelTemporalResidual`. It uses synthetic `(B, T, V, J, 3)` inputs and a fixed 14-camera rig, measures CPU (B=1) and GPU (B=1,4,8,16) configurations, and writes JSON + Markdown reports.
- `experiments/benchmark_inference_v3.py:1–273` — earlier benchmark for the non-temporal `RayAttentionFusionModelV3`; includes an ONNX-export attempt and CPU/GPU throughput tables.
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py:109–185` — `forward()` of the current best model (residual refinement head).
- `motionflow_mv/fusion/ray_attention_temporal_model.py:169–258` — base temporal model with per-frame view/joint attention and temporal transformer.
- `docs/paper_draft_icra_cvpr_2027.md:106–114` — current paper runtime table: 78.3 ms/clip at B=1, up to 195 clips/s at B=16 on RTX 4090.
- `outputs/benchmark_residual_temporal_4090.md` and `outputs/benchmark_residual_temporal_4090.json` — existing results for `d=64`, `residual_hidden=128`, `V=14`, `J=28`, `T=13`.

**Existing numbers**

| Batch | Latency (ms) | Throughput (clips/s) |
|---:|---:|---:|
| 1 | 78.3 | 12.8 |
| 4 | 71.0 | 56.4 |
| 8 | 78.1 | 102.5 |
| 16 | 82.1 | 194.8 |

The model has **243 428** parameters. No `torch.compile`, TensorRT, ONNX, or per-component profiling has been applied yet.

## 2. Gap / opportunity

The current benchmark is architecture-level (random weights) and only reports end-to-end latency. For ICRA/CVPR 2027 we need:

1. **Real-weight, real-data timing** on the final checkpoint (`outputs/ray_attention_temporal_residual_final5.pth`) and a representative WebBridge/MPI-INF-3DHP clip, because synthetic inputs may hide data-movement and preprocessing costs.
2. **Per-component profiling** to identify the bottleneck (temporal transformer vs. DLT vs. residual MLP) and justify architecture decisions.
3. **Deployment-optimized runtimes** via `torch.compile` (available in torch 2.4.0), and a clean separation of the neural network from the geometric DLT step for future ONNX/TensorRT export.
4. **Memory footprint** at batch size 16 to confirm RTX 4090 (24 GB) headroom.

## 3. Concrete next step

Add a new profiling script: `experiments/benchmark_residual_temporal_rtx4090_profile.py`.

It should:

1. Load `outputs/ray_attention_temporal_residual_final5.pth` (or the H36M checkpoint).
2. Run on real multi-view clips from `data/webbridge/mpi_inf_3dhp/` (fallback to synthetic if not present).
3. Benchmark the same batch grid `[1, 4, 8, 16]` and emit `outputs/benchmark_residual_temporal_rtx4090_profile.json/.md`.
4. Use `torch.profiler` to record per-op time for the `forward()` pass at B=1 and B=16.
5. Try `torch.compile(model, mode='default')` and re-run; report whether it lowers latency or fails (the DLT `torch.linalg.lstsq` step may need to be excluded from the compiled region).
6. Record peak GPU memory via `torch.cuda.max_memory_allocated()` and per-batch memory.

The script should reuse the timing logic in `experiments/benchmark_residual_temporal.py` but wrap the model call with `torch.profiler` and `torch.cuda` memory hooks; no changes to core fusion code are required.

## 4. Expected success metric

After running the new script, we will know:

- Updated RTX 4090 end-to-end latency/throughput table on the real final checkpoint.
- A per-component breakdown showing whether the temporal transformer, DLT, or residual MLP dominates latency.
- Whether `torch.compile` improves B=1 latency by ≥5–10% (or fails and needs the DLT split out).
- Peak GPU memory at B=16, confirming the model fits comfortably within 24 GB.
- A markdown report that can replace the placeholder table in `docs/paper_draft_icra_cvpr_2027.md:106–114`.

## 5. Risks / blockers

- **A800-D and Docker are read-only** — do not modify anything there; run only on the local RTX 4090/WSL environment.
- **WebBridge data may need download**; do not commit large files. Use the local `data/webbridge/` cache if present, otherwise fall back to synthetic data for the benchmark numbers.
- **`torch.compile` may fail** on the differentiable DLT `torch.linalg.lstsq` call. If so, split the model into a compiled network (up to per-view weights) and a separate geometric triangulation step.
- **Profiling overhead** from `torch.profiler` can distort latency; run profiler only for the breakdown, not the headline throughput numbers.
- **PyTorch / CUDA 12.1 compatibility** — torch 2.4.0+cu121 is pinned; `torch.compile` should work, but verify no custom kernels break.
