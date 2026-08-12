# Session Summary — 2026-08-12

> Generated at the end of the 2026-08-12 work session.  
> Local repo: `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> Remote A800-D / Docker `motionflow`: **read-only** (no writes, starts, or modifications).

---

## 1. GPU / Resource Status

- **A800 GPU 7**: v85 random-view-dropout training is **RUNNING** (PID `2058225`, Epoch 5 in progress, loss falling). **Do not touch.**
- **A800 GPU 6**: v85 split-k no-fallback variable-view eval is **RUNNING** (k=2 done; k=3/4 in progress). **Do not touch.**
- **A800 disk**: `/mnt/nvme0n1p1` is **~99% full** (~58 GB free).
- **Local RTX 4090**: Idle; reserved for smoke tests only.
- **GPU policy**: Only GPUs 6/7 may be used by this project; GPUs 0–5 are reserved.

---

## 2. Last Two Swarms Accomplished

### 2.1 Swarm A — CVPR 2027 Data-Foundation & v85 Sparse-View Run

**Goal:** Close the true-GT data foundation, quantify sparse-view robustness, and prepare the v85 random-view-dropout experiment.

**Key accomplishments:**

- **H36M true-GT protocol is canonical.** Old circular labels (`data/h36m_hf/`, `data/webbridge/h36m*.npz`) are deprecated; configs referencing them are moved to `configs/deprecated/circular/`. The standard protocol uses `data/h36m_true_gt/` with manifest `configs/splits/h36m_true_gt_standard.yaml`.
- **v85 random-view-dropout training launched on A800 GPU 7.** Epoch 4 val_MPJPE **36.97 mm**; Epoch 5 in progress. This is the first model trained natively on k=2/3/4 via random view dropout (dropout prob 0.3, min 2 views) with active-view-count embedding.
- **v85 no-fallback variable-view eval launched on GPU 6.** Split-k k=2 completed: S9 **2310.27 mm**, S11 **2308.80 mm** (learned model alone still fails catastrophically at k=2, as expected); k=3/4 in progress.
- **v81/v82/v25 DLT-fallback variable-view evals completed.** Fallback numbers baseline: S9 k=2/3/4 = 58.18/33.32/116.98 mm; S11 = 49.35/25.28/110.58 mm.
- **MPI-INF-3DHP real detected 2D completed.** 16/16 `.npz` files in `data/webbridge/mpi_inf_3dhp_detected_2d/`; DLT baseline MPJPE **115.09 mm**, PA-MPJPE **132.68 mm**.
- **AIST++ → H36M cross-eval completed.** Combined MPJPE **~93.94 mm** (S9 98.17 mm, S11 89.70 mm).
- **True-GT leaderboard updated.** Iskakov 23.40 mm; conf-DLT 25.67 mm; v25 stability 31.56 mm; v81 37.83 mm; v82 39.46 mm.

**Artifacts:** `docs/handoff_qwen3.8max.md`, `docs/next_24h_plan_2026-08-12.md`, `docs/results_true_gt_h36m.md`, `docs/cvpr2027_status.md`.

### 2.2 Swarm B — Runtime-Efficiency Benchmark

**Goal:** Refresh real-time efficiency numbers and identify paper-ready gaps before submission.

**Key accomplishments:**

- Ran local RTX 4090 smoke benchmarks on two existing scripts:
  - `experiments/benchmark_runtime.py` — single-frame/clip latency, peak memory, batch throughput for three model variants.
  - `experiments/benchmark_residual_temporal.py` — end-to-end latency/throughput grid for `RayAttentionFusionModelTemporalResidual`.
- **Key local numbers (RTX 4090, torch 2.7.1+cu118, synthetic inputs):**

| Model | Params | Single-frame (ms) | Batch throughput (fps) | Peak mem (MB) |
|---|---|---:|---:|---:|
| `RayAttentionFusionModelV3` | 134,497 | 86.98 | 45.02 | 10.65 |
| `RayAttentionFusionModelTemporal` | 217,825 | 88.53 | 575.15 | 24.90 |
| `RayAttentionFusionModelTemporalResidual` | 243,428 | 87.20 | 596.17 | 25.00 |

- **Gaps flagged vs. paper draft:** Current RTX 4090 numbers are slightly worse than the paper draft table, likely due to PyTorch/CUDA version and lack of `torch.compile`/TensorRT. Real-weight/real-data timing, per-component profiling, and A800 inference numbers are still missing.
- **Recommended next phases:** Phase A (real-weight, real-data RTX 4090 refresh); Phase B (per-component profiling with `torch.profiler`); Phase E (A800 inference benchmark once GPU 6/7 are free).

**Artifacts:** `docs/swarm_iter_next/runtime_benchmark_plan.md`, `docs/swarm_iter_next/runtime_benchmark_report.md`, `outputs/runtime_benchmark_20260812_124934.json`, `outputs/benchmark_residual_temporal_4090.{json,md}`.

---

## 3. Next 3 Concrete Tasks

1. **Wait for v85 to finish and evaluate sparse-view robustness.**
   - Monitor `outputs/ablations/v85_random_view_dropout_medium_a800.log` (GPU 7).
   - Monitor `outputs/variable_view_v85_random_view_dropout_medium_a800_k{2,3,4}.log` (GPU 6).
   - The post-training eval suite monitor (PID `2072251`) will auto-launch test-set eval, fresh no-fallback variable-view eval, and DLT-fallback eval once training completes.
   - Compare v85 k=2/3/4 MPJPE against the v25 DLT-fallback baseline (S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm).
   - If k<4 remains catastrophic, design stronger count-conditioning or a dedicated sparse-view head.

2. **Run A800 disk cleanup dry-run and free space safely.**
   - Run `scripts/cleanup_a800_safe.sh --dry-run` (or similar safe dry-run command in the A800 repo).
   - Identify removable stale outputs (e.g., failed v83/v84 runs, duplicate manifests, old `.failed_*` logs) before launching SOTA baselines.
   - Target: free ≥2 GB without touching running v85 jobs.

3. **Prepare SOTA comparison configs and validate paper numbers.**
   - VoxelPose / MVPose / DLT configs are ready in `scripts/sota_baselines/` and `configs/`; prepare them for launch once GPU 6/7 are free after the v85 suite.
   - Ensure `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md` use the latest non-circular true-GT and v85 numbers.
   - Verify MPI DLT baseline (115.09 mm) and AIST++ cross-eval (~93.94 mm) are correctly recorded.

---

## 4. Files Created / Touched Today

- `docs/session_summary_20260812.md` (this file)
- `docs/swarm_iter_next/runtime_benchmark_plan.md`
- `docs/swarm_iter_next/runtime_benchmark_report.md`
- `docs/handoff_qwen3.8max.md`
- `docs/next_24h_plan_2026-08-12.md`
- `outputs/runtime_benchmark_20260812_124934.json`
- `outputs/benchmark_residual_temporal_4090.json`
- `outputs/benchmark_residual_temporal_4090.md`

---

## 5. Reminders

- **Do not stop, kill, or restart v85 training (GPU 7) or v85 no-fallback eval (GPU 6).**
- **Use only GPU 6/7** for any future A800 work; GPUs 0–5 are reserved.
- **A800 / Docker `motionflow` is read-only** — inspect only, do not write.
- **Local RTX 4090** is for smoke/diagnostics only; avoid starting long training jobs.
