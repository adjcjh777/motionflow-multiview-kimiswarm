"""Profile the cross-view residual + principal-point model with PyTorch profiler.

Usage:
    python experiments/profile_pp_model.py --device cpu --batch_size 1 --clip_len 13
    python experiments/profile_pp_model.py --device cuda --batch_len 13

Only a single forward pass is profiled after a short warmup.  Memory and latency
numbers are printed to stdout; a Chrome trace is written to
``outputs/profile_pp_model_<device>.json`` when ``--export_trace`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def _make_circular_cameras(n_views: int = 4) -> list[Camera]:
    """Build a synthetic circular camera rig."""
    cameras: list[Camera] = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / (np.linalg.norm(c) + 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t_vec = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t_vec))
    return cameras


def _build_model(
    j: int,
    d: int,
    n_views: int,
    n_st_layers: int,
    residual_hidden: int,
    device: torch.device,
) -> RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint:
    """Instantiate the PP model in eval mode on the requested device."""
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=d,
        n_views=n_views,
        n_st_layers=n_st_layers,
        residual_hidden=residual_hidden,
    )
    model.to(device)
    model.eval()
    return model


def _make_input(
    batch_size: int,
    clip_len: int,
    n_views: int,
    j: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random 2D keypoints + confidences and camera tensors."""
    rng = np.random.default_rng(2024)
    points_2d = rng.uniform(0, 640, size=(batch_size, clip_len, n_views, j, 2)).astype(np.float32)
    confidences = rng.uniform(0.5, 1.0, size=(batch_size, clip_len, n_views, j)).astype(np.float32)
    x = np.concatenate([points_2d, confidences[..., None]], axis=-1)  # (B, T, V, J, 3)
    x_tensor = torch.from_numpy(x).to(device)

    cameras = _make_circular_cameras(n_views)
    K = torch.stack([torch.from_numpy(cam.K) for cam in cameras], dim=0).float().to(device)
    R = torch.stack([torch.from_numpy(cam.R) for cam in cameras], dim=0).float().to(device)
    t = torch.stack([torch.from_numpy(cam.t) for cam in cameras], dim=0).float().to(device)

    return x_tensor, K, R, t


def profile_latency(
    model: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
    x: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    n_warmup: int = 10,
    n_iter: int = 50,
) -> dict[str, float]:
    """Return mean / std / p99 latency over ``n_iter`` eval forward passes (ms)."""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, K=K, R=R, t=t)
        if x.device.type == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(n_iter):
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x, K=K, R=R, t=t)
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    times_array = np.array(times)
    return {
        "mean_ms": float(times_array.mean()),
        "std_ms": float(times_array.std(ddof=1)),
        "p50_ms": float(np.median(times_array)),
        "p99_ms": float(np.percentile(times_array, 99)),
        "min_ms": float(times_array.min()),
        "max_ms": float(times_array.max()),
    }


def profile_with_torch_profiler(
    model: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
    x: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    export_trace: bool,
    device_name: str,
) -> dict[str, float]:
    """Run one profiled forward pass and optionally export a Chrome trace."""
    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("model_forward"):
            with torch.no_grad():
                _ = model(x, K=K, R=R, t=t)

    if export_trace:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        trace_path = out_dir / f"profile_pp_model_{device_name}.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"Chrome trace written to {trace_path}")

    stats = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    print("\nTop 10 ops by CPU time:")
    print(stats)

    total_cpu_time_ms = sum(evt.cpu_time_total for evt in prof.events()) / 1000.0
    total_cuda_time_ms = (
        sum(evt.cuda_time_total for evt in prof.events()) / 1000.0
        if x.device.type == "cuda"
        else 0.0
    )

    return {
        "profiler_cpu_time_total_ms": float(total_cpu_time_ms),
        "profiler_cuda_time_total_ms": float(total_cuda_time_ms),
    }


def run_profile(device_name: str, args: argparse.Namespace) -> dict:
    """Profile on a single device and return a metrics dictionary."""
    print(f"\n{'=' * 60}")
    print(f"Profiling on {device_name.upper()}")
    print(f"{'=' * 60}")

    device = torch.device(device_name)
    n_views = args.n_views
    j = args.j
    clip_len = args.clip_len
    batch_size = args.batch_size

    model = _build_model(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        device=device,
    )

    x, K, R, t = _make_input(batch_size, clip_len, n_views, j, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f} M")
    print(f"Input shape: {tuple(x.shape)} (B={batch_size}, T={clip_len}, V={n_views}, J={j})")

    mem_allocated_before = 0.0
    mem_reserved_before = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        mem_allocated_before = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved_before = torch.cuda.memory_reserved(device) / 1024**2

    latency = profile_latency(model, x, K, R, t, n_warmup=args.n_warmup, n_iter=args.n_iter)
    print("\nLatency (ms):")
    for k, v in latency.items():
        print(f"  {k}: {v:.3f}")

    profiler_metrics = profile_with_torch_profiler(
        model, x, K, R, t, export_trace=args.export_trace, device_name=device_name
    )

    metrics: dict[str, float | int | str] = {
        "device": device_name,
        "batch_size": batch_size,
        "clip_len": clip_len,
        "n_views": n_views,
        "j": j,
        "total_params": total_params,
        **latency,
        **profiler_metrics,
    }

    if device.type == "cuda":
        mem_allocated_after = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved_after = torch.cuda.memory_reserved(device) / 1024**2
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        gpu_mem = {
            "gpu_allocated_before_mb": mem_allocated_before,
            "gpu_allocated_after_mb": mem_allocated_after,
            "gpu_reserved_before_mb": mem_reserved_before,
            "gpu_reserved_after_mb": mem_reserved_after,
            "gpu_peak_allocated_mb": peak_allocated,
            "gpu_peak_reserved_mb": peak_reserved,
        }
        print("\nGPU memory (MB):")
        for k, v in gpu_mem.items():
            print(f"  {k}: {v:.2f}")
        metrics.update(gpu_mem)

    cpu_mem_mb = 0.0
    try:
        import psutil

        process = psutil.Process()
        cpu_mem_mb = process.memory_info().rss / 1024**2
        metrics["cpu_rss_mb"] = cpu_mem_mb
        print(f"\nCPU RSS (this process): {cpu_mem_mb:.2f} MB")
    except Exception:
        pass

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the cross-view residual + PP model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to profile on. CUDA is only used if available.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for profiling.")
    parser.add_argument("--clip_len", type=int, default=13, help="Temporal clip length.")
    parser.add_argument("--n_views", type=int, default=14, help="Number of camera views.")
    parser.add_argument("--j", type=int, default=28, help="Number of joints.")
    parser.add_argument("--d", type=int, default=64, help="Feature dimension.")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Number of spatio-temporal layers.")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size.")
    parser.add_argument("--n_warmup", type=int, default=10, help="Warmup iterations before timing.")
    parser.add_argument("--n_iter", type=int, default=50, help="Timed iterations for latency.")
    parser.add_argument("--export_trace", action="store_true", help="Export Chrome trace JSON.")
    parser.add_argument("--json_out", type=str, default=None, help="Path to write metrics JSON.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    metrics = run_profile(args.device, args)

    json_out = args.json_out or f"outputs/profile_pp_model_{args.device}.json"
    out_path = Path(json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {out_path}")


if __name__ == "__main__":
    main()
