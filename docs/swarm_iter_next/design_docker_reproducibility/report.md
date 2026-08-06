# Docker Reproducibility & A800 Benchmark Plan

## Goal

Provide a fully containerised, reproducible runtime for the MotionFlow-MultiView training and benchmarking pipeline, plus an dedicated NVIDIA A800 benchmark script that mirrors the existing RTX 4090 benchmark.

## Scope

This task does **not** modify any model code. It adds:

1. `docs/swarm_iter_next/design_docker_reproducibility/Dockerfile` — container image definition.
2. `docs/swarm_iter_next/design_docker_reproducibility/build.sh` — build the image.
3. `docs/swarm_iter_next/design_docker_reproducibility/run.sh` — start an container with repo mounts.
4. `docs/swarm_iter_next/design_docker_reproducibility/run_benchmark.sh` — run the A800 benchmark inside the container.
5. `scripts/benchmark_a800.py` — A800-focused throughput/latency benchmark.

## Base Image Choice

The project currently pins `torch==2.4.0+cu121` (see `requirements.txt`). The Dockerfile therefore uses `nvidia/cuda:12.1.0-devel-ubuntu22.04` as the runtime base and installs Python 3.10, the pinned PyTorch wheel, and project requirements.

## Mounting Strategy

- Code is bind-mounted from the host repo root to `/workspace/motionflow-multiview` so iterative development does not require image rebuilds.
- Data is expected under `./data` and is bind-mounted read-write.
- Outputs are written to `./outputs` on the host.

## Reproducibility Measures

- Fixed CUDA 12.1 / cuDNN via base image.
- Fixed `torch==2.4.0+cu121` in `requirements.txt`.
- `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, and deterministic flags set in the benchmark script.
- No host conda environment is required inside the container.

## A800 Benchmark Script

`scripts/benchmark_a800.py` reuses `RayAttentionFusionModelTemporalResidual` and the same synthetic rig pattern as `experiments/benchmark_residual_temporal.py`, but:

- Targets the A800 (80 GB) with larger batch sizes (`1, 8, 16, 32, 64`).
- Reports memory utilisation.
- Emits JSON + Markdown under `outputs/benchmark_a800/`.
- Defaults to `float32` but can switch to `float16` for a mixed-precision run.

## How to Use

```bash
# 1. Build image
bash docs/swarm_iter_next/design_docker_reproducibility/build.sh

# 2. Launch interactive container
bash docs/swarm_iter_next/design_docker_reproducibility/run.sh

# 3. Inside container, run A800 benchmark
bash /workspace/motionflow-multiview/docs/swarm_iter_next/design_docker_reproducibility/run_benchmark.sh
```

Or, on a bare-metal A800 node with the repo already cloned:

```bash
python scripts/benchmark_a800.py --batch_sizes 1 8 16 32 64 --iters 200
```

## Expected Impact

- Enables one-command reproduction of the training/evaluation environment on A800 clusters.
- Provides a standardised A800 throughput baseline for paper tables and CI reports.
- Makes it easy for reviewers and collaborators to rerun the benchmark without configuring conda/CUDA manually.

## Notes / Blockers

- The Dockerfile assumes the host has NVIDIA Docker runtime (`nvidia-docker2`) and an A800-capable driver (>= 525).
- Dataset files are not included in the image and must be mounted at runtime.
