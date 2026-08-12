# Runtime Benchmark Plan (MotionFlow-MultiView)

> Status: local WSL smoke numbers refreshed on 2026-08-12. Plan only; no A800 or training jobs launched.

## 1. What was checked

I inspected the existing runtime-benchmark scripts and ran two synthetic smoke benchmarks on the local WSL RTX 4090 environment:

| Script | Purpose | Output |
|--------|---------|--------|
| `experiments/benchmark_runtime.py` | Single-frame latency, clip latency, peak memory, batch throughput for three model variants | `outputs/runtime_benchmark_20260812_123407.json`, `docs/swarm_iter_next/runtime_benchmark_report.md` |
| `experiments/benchmark_residual_temporal.py` | End-to-end latency/throughput grid for `RayAttentionFusionModelTemporalResidual` | `outputs/benchmark_residual_temporal_4090.json`, `outputs/benchmark_residual_temporal_4090.md` |

Environment:

```text
Python 3.13.9
torch 2.7.1+cu118
CUDA available: True
Device: cuda (RTX 4090)
```

## 2. Exact commands and outputs

### 2.1 `experiments/benchmark_runtime.py`

```bash
python experiments/benchmark_runtime.py --iters 100 --warmup 20
```

Results (`outputs/runtime_benchmark_20260812_123407.json`):

| Model | Params | Single-frame (ms) | Clip (ms) | Batch throughput (fps) | Peak mem (MB) | 60 FPS? | 30 FPS? |
|-------|--------:|------------------:|----------:|------------------------:|--------------:|--------:|--------:|
| `RayAttentionFusionModelV3` | 134,497 | 85.12 | 85.12 | 47.19 | 10.65 | No | No |
| `RayAttentionFusionModelTemporal` | 217,825 | 85.99 | 86.19 | 571.66 | 24.90 | No | No |
| `RayAttentionFusionModelTemporalResidual` | 243,428 | 90.07 | 88.85 | 570.85 | 25.00 | No | No |

### 2.2 `experiments/benchmark_residual_temporal.py`

```bash
python experiments/benchmark_residual_temporal.py --iters 100 --warmup 20
```

Results (`outputs/benchmark_residual_temporal_4090.json`):

| Batch | Latency (ms) | Throughput (clips/s) | Total frames |
|-------|-------------:|-----------------:|-------------:|
| 1 | 86.31 | 11.59 | 100 |
| 4 | 90.23 | 44.33 | 400 |
| 8 | 95.83 | 83.48 | 800 |
| 16 | 91.99 | 173.93 | 1600 |

CPU (B=1): 80.76 ms latency, 12.38 clips/s.

### 2.3 Comparison with paper draft

The current paper draft (`docs/paper_draft_icra_cvpr_2027.md`, Section 5.7) reports:

| Batch | Latency (ms) | Throughput (clips/s) |
|---:|---:|---:|
| 1 | 78.3 | 12.8 |
| 4 | 71.0 | 56.4 |
| 8 | 78.1 | 102.5 |
| 16 | 82.1 | 194.8 |

Our fresh local run shows slightly worse latency/throughput, likely due to a different PyTorch/CUDA version (2.7.1+cu118 vs. the draft's 2.4.0+cu121), background load, and the lack of `torch.compile`/TensorRT. The headline claim should be re-verified on the same target hardware and software stack before submission.

## 3. Gaps identified

1. **Real-weight, real-data timing.** Both existing scripts use synthetic inputs and random weights. The actual checkpoint (`outputs/ray_attention_temporal_residual_final5.pth` or the current best v25/v81/v82/v85 checkpoint) and a representative MPI-INF-3DHP/H36M clip may have different data-movement and preprocessing costs.
2. **Per-component profiling.** No breakdown exists for the temporal transformer vs. DLT `torch.linalg.lstsq` vs. residual MLP, so we cannot justify where latency spends its budget.
3. **Deployment optimizations.** `torch.compile`, TensorRT, and ONNX export are untapped. The DLT step (`torch.linalg.lstsq`) is a known risk for compilation.
4. **Memory headroom.** Peak memory at B=16 is unmeasured on the real model; the lightweight synthetic test reports only ~25 MB, which is far below the RTX 4090 24 GB limit but not representative of full inference with gradients disabled and real data.
5. **A800 server numbers.** Only local RTX 4090 numbers exist. For a CVPR 2027 submission, A800 inference throughput and batch scalability are useful supplementary data, especially for the variable-view eval pipeline.
6. **Variable-view / sparse-view runtime.** k=2/3/4 runtime behavior has not been benchmarked, even though sparse-view robustness is a paper pillar.

## 4. Recommended plan

### Phase A: Refresh paper-ready RTX 4090 numbers (local WSL, no A800)

1. Create `experiments/benchmark_residual_temporal_rtx4090_profile.py` that:
   - Loads the current best checkpoint (v25 stability or later v81/v82/v85 once ready).
   - Runs on real data when available (H36M true-GT test clip, MPI-INF-3DHP detected 2D), with synthetic fallback.
   - Repeats the B=[1,4,8,16] grid with 100+ iterations and warmup.
   - Records `torch.cuda.max_memory_allocated()` per batch size.
   - Writes `outputs/benchmark_residual_temporal_rtx4090_profile.json/.md`.
2. Verify/refresh the paper draft table in `docs/paper_draft_icra_cvpr_2027.md:364-369` with the new numbers.

### Phase B: Per-component profiling (local WSL)

1. Wrap `RayAttentionFusionModelTemporalResidual.forward()` with `torch.profiler` at B=1 and B=16.
2. Report time spent in:
   - ray/view attention block,
   - temporal transformer,
   - DLT/triangulation (`torch.linalg.lstsq`),
   - residual MLP/refinement head.
3. Identify the dominant bottleneck; if it is the DLT, document that geometric triangulation should be split from the compiled neural network for deployment.

### Phase C: Deployment optimization sweep (local WSL)

1. Benchmark `torch.compile(model, mode="default")` and `mode="reduce-overhead"`.
2. If compilation fails on the DLT step, split the model into:
   - compiled neural network (up to per-view weights),
   - standalone geometric triangulation step.
3. Optionally evaluate ONNX/TensorRT export for the compiled neural network only.
4. Target: ≥5-10% latency reduction on B=1 without changing accuracy.

### Phase D: Sparse-view / variable-view runtime (local WSL)

1. Benchmark `RayAttentionFusionModelTemporalResidual` and `HardenedVariableViewInferenceWrapper` at k=2,3,4.
2. Compare learned-model latency vs. direct DLT fallback latency.
3. Document whether view dropout (v85) introduces extra overhead from count embedding or masking.

### Phase E: A800 inference benchmark (run only when GPU 6/7 free)

1. Run `scripts/benchmark_a800.py` on A800 GPU 6 or 7 with B=[1,8,16,32,64] and `--iters 200`.
2. Report throughput scaling and peak memory on the A800 (80 GB).
3. This is explicitly queued behind v85 training/eval; do not launch until GPU 6/7 is free.

## 5. Conclusion / recommendation

- The existing runtime scripts are functional and the model remains lightweight (~243k params, ~25 MB peak memory on synthetic inputs).
- Fresh local numbers are slightly worse than the paper draft table; the draft table should be updated after a controlled run on the target RTX 4090 stack.
- Priority: Phase A (real-weight, real-data refresh) + Phase B (per-component profiling). These are local, low-risk, and directly feed the paper.
- Phase C (torch.compile/ONNX) is a high-value follow-up if the bottleneck is in the neural network rather than the DLT.
- Phase E (A800 benchmark) is deferred until GPU 6/7 is free, per project GPU policy.

## 6. Files touched / created

- `docs/swarm_iter_next/runtime_benchmark_plan.md` (this file)
- `docs/swarm_iter_next/runtime_benchmark_report.md` (auto-generated by `experiments/benchmark_runtime.py`)
- `outputs/runtime_benchmark_20260812_123407.json`
- `outputs/benchmark_residual_temporal_4090.json`
- `outputs/benchmark_residual_temporal_4090.md`

No training or evaluation jobs were launched on A800; all work was local synthetic benchmarking only.
