"""Real-time efficiency benchmark for MotionFlow-MultiView fusion models.

Measures end-to-end single-frame (B=1) latency, throughput, memory footprint,
and real-time feasibility for the key model variants on CPU and CUDA. Synthetic
2D keypoints and a fixed camera rig are used so the script is data-agnostic and
runs without dataset dependencies.

Usage (from repo root):
    python experiments/benchmark_runtime.py

Outputs:
    outputs/runtime_benchmark_<timestamp>.json
    docs/swarm_iter_next/runtime_benchmark_report.md
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3
from motionflow_mv.fusion.ray_attention_temporal_model import RayAttentionFusionModelTemporal
from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


class _FixedRig:
    """Single fixed camera rig built with pure torch/math (no NumPy BLAS)."""

    def __init__(self, V: int, device: torch.device):
        self.V = V
        self.K, self.R, self.t = self._build_rig(device)

    @staticmethod
    def _rotation_y(angle: float) -> torch.Tensor:
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32
        )

    def _build_rig(self, device: torch.device):
        V = self.V
        Ks, Rs, ts = [], [], []
        radius = 5.0
        for i in range(V):
            fx = fy = 1000.0 + (i % 3) * 50.0
            cx, cy = 640.0, 360.0
            K = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
            )
            Ks.append(K)

            angle = 2.0 * math.pi * i / V
            Rs.append(self._rotation_y(angle))

            t_vec = torch.tensor(
                [math.cos(angle) * radius, 0.0, math.sin(angle) * radius],
                dtype=torch.float32,
            )
            t_vec = t_vec / (t_vec.norm() + 1e-8) * radius
            ts.append(t_vec)

        K = torch.stack(Ks)
        R = torch.stack(Rs)
        t = torch.stack(ts)
        return K.to(device), R.to(device), t.to(device)

    def to(self, device: torch.device):
        return self.K.to(device), self.R.to(device), self.t.to(device)


def _dummy_batch(batch_size: int, T: int, V: int, J: int, device: torch.device):
    x = torch.randn(batch_size, T, V, J, 3, device=device)
    x[..., 2] = torch.sigmoid(x[..., 2])
    return x


def _warmup_and_benchmark(model, x, K, R, t, device, warmup, iters):
    """Return average latency in seconds over ``iters`` forward passes."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, K=K, R=R, t=t)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = model(x, K=K, R=R, t=t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / iters


def _measure_memory(model, x, K, R, t, device):
    """Return peak GPU memory in MB for a single forward pass, or None on CPU."""
    if device.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(x, K=K, R=R, t=t)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024.0 ** 2)


def _is_temporal(model):
    """Return True if the model accepts 5D (B, T, V, J, 3) clip inputs."""
    return hasattr(model, "temporal_attn")


def benchmark_model(model_name, model, device, V, J, clip_len, warmup, iters, rig):
    """Benchmark a single model variant under real-time-relevant conditions."""
    model = model.to(device)
    model.eval()
    K, R, t = rig.to(device)
    temporal = _is_temporal(model)

    # Single-frame B=1 latency (real-time streaming scenario).
    # V3 expects (B, V, J, 3); temporal models accept a single-frame clip (B, 1, V, J, 3).
    if temporal:
        x_single = _dummy_batch(1, 1, V, J, device)
    else:
        x_single = _dummy_batch(1, 1, V, J, device).squeeze(1)  # (B, V, J, 3)
    single_latency = _warmup_and_benchmark(model, x_single, K, R, t, device, warmup, iters)

    if temporal:
        # Clip B=1 latency (temporal models).
        x_clip = _dummy_batch(1, clip_len, V, J, device)
        clip_latency = _warmup_and_benchmark(model, x_clip, K, R, t, device, warmup, iters)

        # Peak memory on the clip input.
        peak_mem_mb = _measure_memory(model, x_clip, K, R, t, device)

        # Small-batch throughput (B=4) on clips.
        x_batch = _dummy_batch(4, clip_len, V, J, device)
        batch_latency = _warmup_and_benchmark(model, x_batch, K, R, t, device, warmup, iters)
        batch_total_frames = 4 * clip_len * iters
        throughput_fps = batch_total_frames / (batch_latency * iters)
    else:
        # Non-temporal model: clip metrics mirror single-frame, throughput is B=4 single-frame.
        x_clip = x_single
        clip_latency = single_latency
        peak_mem_mb = _measure_memory(model, x_clip, K, R, t, device)

        x_batch = _dummy_batch(4, 1, V, J, device).squeeze(1)
        batch_latency = _warmup_and_benchmark(model, x_batch, K, R, t, device, warmup, iters)
        batch_total_frames = 4 * iters
        throughput_fps = batch_total_frames / (batch_latency * iters)

    return {
        "model": model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "single_latency_ms": single_latency * 1000.0,
        "single_fps": 1.0 / single_latency,
        "clip_latency_ms": clip_latency * 1000.0,
        "clip_fps": 1.0 / clip_latency,
        "batch_throughput_fps": throughput_fps,
        "peak_memory_mb": peak_mem_mb,
    }


def _build_model(name, V, J, d, residual_hidden):
    if name == "RayAttentionFusionModelV3":
        return RayAttentionFusionModelV3(j=J, d=d, n_views=V, n_heads=4, n_joint_layers=1)
    if name == "RayAttentionFusionModelTemporal":
        return RayAttentionFusionModelTemporal(
            j=J, d=d, n_views=V, n_heads=4, n_joint_layers=1, n_temporal_layers=2
        )
    if name == "RayAttentionFusionModelTemporalResidual":
        return RayAttentionFusionModelTemporalResidual(
            j=J, d=d, n_views=V, n_heads=4, n_joint_layers=1,
            n_temporal_layers=2, residual_hidden=residual_hidden,
        )
    raise ValueError(f"Unknown model: {name}")


def _real_time_status(latency_ms: float):
    """Return whether the measured latency can meet common real-time targets."""
    # Use a 50 % slack on the frame budget to account for preprocessing / I/O.
    budget_60hz = 1000.0 / 60.0 * 0.5
    budget_30hz = 1000.0 / 30.0 * 0.5
    return {
        "meets_60fps_streaming": latency_ms <= budget_60hz,
        "meets_30fps_streaming": latency_ms <= budget_30hz,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_views", type=int, default=14)
    parser.add_argument("--j", type=int, default=28)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    models = args.models or [
        "RayAttentionFusionModelV3",
        "RayAttentionFusionModelTemporal",
        "RayAttentionFusionModelTemporalResidual",
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    report_dir = Path("docs/swarm_iter_next")
    report_dir.mkdir(exist_ok=True)

    V, J, T = args.n_views, args.j, args.clip_len
    rig = _FixedRig(V, torch.device("cpu"))

    results = []
    print(f"Benchmarking on {device} ...")
    for name in models:
        print(f"  - {name}")
        model = _build_model(name, V, J, args.d, args.residual_hidden)
        result = benchmark_model(name, model, device, V, J, T, args.warmup, args.iters, rig)
        result.update(_real_time_status(result["single_latency_ms"]))
        results.append(result)
        print(f"       params={result['params']:,}  "
              f"single={result['single_latency_ms']:.2f} ms  "
              f"clip={result['clip_latency_ms']:.2f} ms  "
              f"throughput={result['batch_throughput_fps']:.1f} fps")

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "settings": {
            "n_views": V,
            "joints": J,
            "clip_len": T,
            "d": args.d,
            "residual_hidden": args.residual_hidden,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "results": results,
    }

    json_path = out_dir / f"runtime_benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # Markdown report for docs/swarm_iter_next.
    md_lines = [
        "# Real-Time Efficiency Benchmark Report",
        "",
        f"**Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  ",
        f"**Device**: `{device}`  ",
        f"**Input shape**: (B, T={T}, V={V}, J={J}, 3)  ",
        f"**Warmup iterations**: {args.warmup}  ",
        f"**Measured iterations**: {args.iters}  ",
        "",
        "## Model Variants",
        "",
        "| Model | Params | Single-frame (ms) | Single FPS | Clip (ms) | Throughput (fps) | Peak Mem (MB) | 60 FPS? | 30 FPS? |",
        "|-------|--------:|------------------:|-----------:|----------:|-----------------:|----------------:|--------:|--------:|",
    ]

    for r in results:
        md_lines.append(
            f"| {r['model']} | {r['params']:,} | "
            f"{r['single_latency_ms']:.2f} | {r['single_fps']:.1f} | "
            f"{r['clip_latency_ms']:.2f} | {r['batch_throughput_fps']:.1f} | "
            f"{r['peak_memory_mb'] if r['peak_memory_mb'] is not None else 'N/A'} | "
            f"{'Yes' if r['meets_60fps_streaming'] else 'No'} | "
            f"{'Yes' if r['meets_30fps_streaming'] else 'No'} |"
        )

    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Single-frame latency**: time to process one `(B=1, T=1)` multi-view frame. "
        "This is the most relevant metric for streaming/real-time deployment.",
        "- **Clip latency**: time to process one `(B=1, T=13)` clip, typical of the "
        "temporal model's inference unit.",
        "- **Batch throughput**: frames per second when processing `(B=4, T=13)` batches; "
        "represents offline / batched throughput rather than streaming latency.",
        "- **Peak memory**: maximum GPU memory allocated during a single `(B=1, T=13)` "
        "forward pass; CPU runs report `N/A`.",
        "",
        "## Real-time feasibility",
        "",
        "A model is marked as meeting a real-time target when its single-frame latency "
        "leaves at least 50 % of the frame budget free for preprocessing, I/O, and "
        "downstream pipeline stages. For 60 Hz streaming the budget is "
        "`1000 / 60 * 0.5 ~ 8.33 ms`; for 30 Hz it is `1000 / 30 * 0.5 ~ 16.67 ms`.",
        "",
        "## Notes",
        "",
        "- This benchmark uses randomly initialized weights; reported numbers reflect "
        "architecture-level latency/throughput and are independent of learned weights.",
        "- Synthetic inputs and a fixed camera rig remove any dataset dependency, so the "
        "script can be run on any CUDA or CPU host as a quick smoke test.",
        "- The `RayAttentionFusionModelTemporalResidual` is the current best model and "
        "therefore the primary target for RTX 4090 real-time evaluation.",
        "",
        "## Methodology and design decisions",
        "",
        "1. **Synthetic fixed rig** - Reuses the pure-torch `_FixedRig` helper from "
        "`experiments/benchmark_residual_temporal.py` so the benchmark is dataset-agnostic "
        "and runs on any host.",
        "2. **Single-frame latency is the real-time gate** - Streaming applications must "
        "process each incoming frame before the next one arrives, so `B=1, T=1` latency "
        "is reported as the primary real-time metric.",
        "3. **Clip and batch metrics for throughput** - Clip latency shows the cost of "
        "temporal models that operate on `(B=1, T=13)` windows, while `B=4` throughput "
        "estimates offline/batched capacity.",
        "4. **Memory headroom** - GPU peak memory is captured via "
        "`torch.cuda.max_memory_allocated` to flag models that may exhaust frame-buffer "
        "budgets on edge devices.",
        "5. **Real-time target with 50 % slack** - A model is considered feasible for a "
        "given frame rate only if it uses at most half of the frame budget, leaving room "
        "for preprocessing, I/O, and downstream stages.",
        "",
        "## Limitations and future work",
        "",
        "- CPU numbers here are for smoke-test verification only; the intended target "
        "hardware is an NVIDIA RTX 4090.",
        "- `torch.compile` and TensorRT/ONNX export are not evaluated in this baseline; "
        "both can materially improve latency and should be benchmarked in follow-up work.",
        "- The per-joint count defaults to 28 and view count to 14, matching the "
        "MPI-INF-3DHP setup used by the current best model.",
    ])

    md_path = report_dir / "runtime_benchmark_report.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
