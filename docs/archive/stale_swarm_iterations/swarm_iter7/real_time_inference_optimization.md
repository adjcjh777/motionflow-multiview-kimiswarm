# Real-Time Inference Optimization

## Problem Statement

MotionFlow-MultiView's current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, is a 0.25 M-parameter transformer that performs a full spatio-temporal attention pass over `(T, V, J)` tokens followed by weighted DLT triangulation. Before committing to costly GPU experiments in factorized attention, SDPA/FlashAttention, or distillation, we need a reproducible latency baseline and a minimal, low-risk optimization probe. This direction is therefore gated on **profiling the current PP model first** and testing the cheapest compiler-level speedup (`torch.compile`) before any architecture changes.

## Simplest Concrete Next Experiment

Run a CPU-safe latency benchmark that:

1. Profiles the eager-mode PP model on a representative clip `(B=1, T=13, V=14, J=28)`.
2. Attempts `torch.compile` with default settings as a zero-code-change optimization.
3. Records mean / p50 / p99 latency, parameter count, and memory footprint.

GPU training is **not** required; the script only runs inference on CPU (CUDA optional).

## Files to Touch

- `experiments/profile_pp_model.py` — existing baseline profiler (already in repo).
- `experiments/profile_pp_optimizations.py` — **new** script that compares baseline vs `torch.compile` (added in this commit).
- `docs/swarm_iter7/real_time_inference_optimization.md` — this report.

### Rough diff / sketch

```text
experiments/profile_pp_optimizations.py   | +250 lines
motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py  | read-only
docs/swarm_iter7/real_time_inference_optimization.md | +60 lines
```

The new script reuses the synthetic camera rig and input generation from `profile_pp_model.py` and adds a second `torch.compile(model, mode="default")` branch. No existing experiment runner is modified, so currently queued GPU jobs are unaffected.

## Commands Run and Results

### Baseline CPU profile

```bash
python experiments/profile_pp_model.py --device cpu --batch_size 1 --clip_len 13 --n_iter 50
```

```text
Model parameters: 0.25 M
Input shape: (1, 13, 14, 28, 3)
Latency (ms):
  mean_ms: 93.263
  std_ms:  18.118
  p50_ms:  87.512
  p99_ms:  161.339
```

### Optimization comparison (CPU)

```bash
python experiments/profile_pp_optimizations.py --device cpu --n_iter 50
```

Result:

```text
variant                       mean      std      p50      p99
baseline_eager              75.674    6.769   73.944   96.178
```

`torch.compile` failed on this Windows/WSL environment:

1. First attempt: source-encoding decode error (GBK). Setting `PYTHONUTF8=1` resolved it.
2. Second attempt: `InvalidCxxCompiler: Compiler: cl is not found.` — a C++ compiler is required for `torch.compile` on CPU, which is not installed.

Consequently, the **baseline is confirmed at ~75–93 ms per clip on CPU**, but the zero-code `torch.compile` shortcut is blocked by the build environment.

## Expected Success Metric

- [x] Reproducible CPU latency baseline for the current PP model: **~75 ms mean / ~73 ms p50 per clip** on CPU.
- [ ] `torch.compile` speedup measured (blocked by missing C++ compiler; revisit on Linux/WSL with `g++` or on CUDA where nvcc is present).
- [ ] Next step: profile on the RTX 4090 once the running curriculum job finishes, and compare against a factorized-attention / SDPA variant if CPU compile cannot be made to work.

## Resource Requirement

- **CPU-only** for the profiling and comparison scripts.
- No GPU training is started.
- No writes to A800-D.
