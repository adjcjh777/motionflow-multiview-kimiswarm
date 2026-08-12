# P19 Real-Time Inference Optimization Plan

**Branch:** `feat/swarm-iter18-omniview`  
**Goal:** Profile the current best model (`RayAttentionFusionModelBayesianTriV2`) and define a concrete optimization roadmap toward real-time, publishable-quality inference for ICRA/CVPR 2027.

## 1. Current state

The current best result (8.35 mm MPJPE on MPI-INF-3DHP S2/Seq1) is produced by an **ensemble of `bayesian_tri_v2` variants**. The production single-model is:

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
- Configuration: `d=128`, `residual_hidden=256`, `n_st_layers=3`, 17 joints, 4 views.
- ~929 k parameters.

Existing benchmarking utilities:

- `experiments/benchmark_runtime.py` — end-to-end latency/throughput for V3/Temporal/TemporalResidual (not BayesianTriV2).
- `experiments/benchmark_inference_v3.py` — V3 latency + ONNX export attempt (fails due to `torch.linalg.lstsq`).
- `experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py` — **new** per-component profiler for the BayesianTriV2 forward.

## 2. Profiling methodology

A new isolated profiler was added at `experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py`. It:

1. Builds an instrumented subclass of `RayAttentionFusionModelBayesianTriV2` that wraps each major block with `time.perf_counter`.
2. Warms up, then averages end-to-end latency over multiple iterations.
3. Reports per-component wall-clock time for the last pass.
4. Uses a synthetic fixed rig so the script is dataset-agnostic and runs on CPU.

The numbers below are **CPU smoke-test baselines**; target deployment is CUDA (RTX 4090 / A800).

## 3. Baseline latency results (CPU)

Run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py \
    --n_st_layers 3 --d 128 --residual_hidden 256 --j 17 --n_views 4 --clip_len 13 --iters 10 --warmup 3
```

| Scenario | Latency | FPS |
|----------|--------:|----:|
| Single-frame (`B=1, T=1`) | 53.55 ms | 18.7 |
| Clip (`B=1, T=13`) | 31.29 ms | 32.0 |

The clip is **faster per frame** because the per-frame encoder and the spatio-temporal transformer amortize over `T`, which is the natural operating mode.

Per-component breakdown for the `T=13` clip:

| Component | ms | % |
|-----------|---:|---:|
| `extract_frame_features` | 9.605 | 32.4 |
| `spatio_temporal_attention` | 7.294 | 24.6 |
| `gauss_newton` | 3.556 | 12.0 |
| `triangulation` | 2.612 | 8.8 |
| `epipolar_loss` | 1.805 | 6.1 |
| `damping_head` | 2.088 | 7.0 |
| `residual_mlp` | 0.886 | 3.0 |
| `covariance_head` | 0.795 | 2.7 |
| `principal_point_correction` | 0.745 | 2.5 |
| `weight_head` | 0.211 | 0.7 |
| `projection_matrix` | 0.086 | 0.3 |
| `visibility_multiplier` | 0.003 | <0.1 |

Raw JSON: `outputs/swarm_iter18/bayesian_tri_v2_profile_20260807_030015.json`

## 4. Bottleneck analysis

1. **Per-frame feature extractor (`extract_frame_features`) — 32.4 %**
   - Contains ray embedding, camera MLP, view-level MHA, and joint-level transformer layers.
   - The view-level `MultiheadAttention` and `joint_attn` layers dominate.

2. **Spatio-temporal transformer (`spatio_temporal_attention`) — 24.6 %**
   - `nn.TransformerEncoderLayer` over `T*V` tokens per joint.
   - Standard PyTorch MHA is memory-bandwidth bound and not optimized for small sequence lengths.

3. **Adaptive Gauss-Newton refinement (`gauss_newton`) — 12.0 %**
   - Differentiable per-joint loop (`num_iters=2`) solving `3x3` linear systems.
   - Each iteration builds Jacobians and calls `torch.linalg.solve`.

4. **Triangulation (`triangulation`) — 8.8 %**
   - `torch.linalg.lstsq` on `(N, J, 2V, 3)` systems.
   - Already fully batched, but still a dense linear algebra op.

5. **Training-only epipolar loss (`epipolar_loss`) — 6.1 %**
   - Not needed at inference; safe to remove for deployment.

## 5. Proposed optimizations

### 5.1 Immediate, low-risk wins

| # | Optimization | Expected gain | Risk |
|---|-------------|---------------|------|
| 1 | **Disable `epipolar_loss` at inference.** | ~6 % | None; only a training loss. |
| 2 | **Reduce Gauss-Newton iterations from 2 → 1 (or 0 in a fast mode).** | ~6–12 % | Minor accuracy drop; validate on MPI-INF-3DHP. |
| 3 | **Run the damping/covariance heads only when needed.** | Small | Already cheap, but can be fused. |
| 4 | **Use `torch.compile` with `mode='reduce-overhead'` on the full forward.** | 1.5–3× on CUDA | Graph breaks on `lstsq`/dynamic shapes; may need `dynamic=False` and fixed `T`. |

### 5.2 Architecture / algorithmic optimizations

| # | Optimization | Expected gain | Risk |
|---|-------------|---------------|------|
| 5 | **Replace standard MHA with FlashAttention / memory-efficient attention.** | 2–4× on attention-heavy blocks | Needs CUDA ≥ Ampere; CPU fallback required. |
| 6 | **Factorized spatio-temporal attention:** separate time-attention and cross-view attention instead of joint `(T*V)` attention. | 2–3× on `spatio_temporal_attention` | Requires re-training; ablation needed. |
| 7 | **Lightweight per-frame encoder:** replace joint-level transformer with graph/conv joint mixer or reduce `n_joint_layers`. | 1.5–2× on `extract_frame_features` | May hurt accuracy; smoke-test on H36M/MPI. |
| 8 | **Custom CUDA DLT kernel** for triangulation and Gauss-Newton to fuse loops and avoid Python overhead. | 2–5× on geometry ops | Implementation effort; validate numerically. |
| 9 | **Early view pruning** using predicted weights to drop low-contribution views before expensive heads. | 10–30 % when views > 4 | Harder with variable views; accuracy gate. |

### 5.3 Deployment-level optimizations

| # | Optimization | Expected gain | Risk |
|---|-------------|---------------|------|
| 10 | **ONNX / TensorRT export of the network up to the weight head**, with triangulation in a custom op. | 2–4× end-to-end | `torch.linalg.lstsq` not in default ONNX opset; custom op needed. |
| 11 | **FP16/BF16 mixed inference** for attention and MLPs. | 1.5–2× | Gauss-Newton currently casts to `float32` for stability; keep that path. |
| 12 | **INT8 quantization** of MLP/attention weights for edge deployment. | 2–3× | Accuracy drop must be measured. |
| 13 | **Streaming temporal window:** keep a fixed history buffer and run only the latest `T` small (e.g., `T=5` instead of `T=13`). | Proportional to `T` | Temporal smoothing trade-off. |

## 6. Target performance

For real-time deployment we propose:

- **30 Hz streaming:** end-to-end latency ≤ 16.67 ms on RTX 4090.
- **60 Hz streaming:** end-to-end latency ≤ 8.33 ms on RTX 4090.

Given that the CPU clip already runs at ~31 ms with no GPU parallelization, these targets are plausible after:

1. Removing `epipolar_loss`.
2. `torch.compile` + FlashAttention.
3. Reducing GN iterations or using a custom geometry kernel.
4. Possibly lowering `T` in streaming mode.

## 7. Suggested execution order

1. **P19.1** — Add an inference-only flag to `RayAttentionFusionModelBayesianTriV2` that skips `epipolar_loss` and optionally reduces GN iterations. Validate no MPJPE regression.
2. **P19.2** — Benchmark the same profiler on an RTX 4090/A800 to establish a CUDA baseline.
3. **P19.3** — Apply `torch.compile` and FlashAttention to the attention blocks; measure per-component speedup.
4. **P19.4** — Prototype a custom CUDA op for batched DLT + Gauss-Newton (or wrap with `torch.utils.cpp_extension`).
5. **P19.5** — Explore ONNX/TensorRT export with a custom triangulation op for edge deployment.
6. **P19.6** — If targets are still not met, train a factorized-attention variant (`n_st_layers` factorization) and compare accuracy/latency.

## 8. Risks and blockers

- **Dynamic shapes:** `torch.compile` may struggle with variable `T` and `V`. Fixing `T` for streaming is the simplest path.
- **ONNX export:** `torch.linalg.lstsq` is not supported. A deployment split (network → custom op) is required.
- **Accuracy vs. latency trade-off:** Reducing GN iterations or attention capacity must be tracked on MPI-INF-3DHP and H36M.
- **Hardware access:** Real profiling requires CUDA; CPU numbers in this plan are directional only.

## 9. Deliverables

- `docs/swarm_iter18/P19_realtime_plan.md` — this document.
- `experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py` — per-component CPU profiler for BayesianTriV2.
- `outputs/swarm_iter18/bayesian_tri_v2_profile_*.json` — example profiling data.

## 10. Conclusion

The largest inference costs are the per-frame encoder, the spatio-temporal transformer, and the Gauss-Newton refinement. A combination of inference-only code paths, `torch.compile`/FlashAttention, and a custom geometry kernel is expected to bring the model well within real-time budgets on CUDA while preserving the 8.35 mm MPJPE-level accuracy. The next concrete step is to run the profiler on target hardware and apply the low-risk wins in P19.1.
