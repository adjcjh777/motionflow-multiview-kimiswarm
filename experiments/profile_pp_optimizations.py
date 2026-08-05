"""Compare inference-time optimizations for the cross-view PP model.

This is a CPU-safe, no-training benchmark.  It compares the baseline eager
forward pass against a ``torch.compile`` wrapped version and, if available,
against the scaled-dot-product attention (SDPA) path.  Results are written as
JSON and printed in a table.

Usage:
    python experiments/profile_pp_optimizations.py --device cpu --n_iter 50
    python experiments/profile_pp_optimizations.py --device cuda --n_iter 50
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random 2D keypoints + confidences and camera tensors."""
    rng = np.random.default_rng(2024)
    points_2d = rng.uniform(0, 640, size=(batch_size, clip_len, n_views, j, 2)).astype(np.float32)
    confidences = rng.uniform(0.5, 1.0, size=(batch_size, clip_len, n_views, j)).astype(np.float32)
    x = np.concatenate([points_2d, confidences[..., None]], axis=-1)
    x_tensor = torch.from_numpy(x).to(device)

    cameras = _make_circular_cameras(n_views)
    K = torch.stack([torch.from_numpy(cam.K) for cam in cameras], dim=0).float().to(device)
    R = torch.stack([torch.from_numpy(cam.R) for cam in cameras], dim=0).float().to(device)
    t = torch.stack([torch.from_numpy(cam.t) for cam in cameras], dim=0).float().to(device)

    return x_tensor, K, R, t


def profile_variant(
    name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    n_warmup: int,
    n_iter: int,
) -> dict[str, float]:
    """Return latency stats for a single model variant."""
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

    times_arr = np.array(times)
    return {
        "variant": name,
        "mean_ms": float(times_arr.mean()),
        "std_ms": float(times_arr.std(ddof=1)),
        "p50_ms": float(np.median(times_arr)),
        "p99_ms": float(np.percentile(times_arr, 99)),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare inference optimizations for the cross-view PP model."
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--n_views", type=int, default=14)
    parser.add_argument("--j", type=int, default=28)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    x, K, R, t = _make_input(
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        n_views=args.n_views,
        j=args.j,
        device=device,
    )

    results: list[dict] = []

    # Baseline eager model.
    model_eager = _build_model(
        j=args.j,
        d=args.d,
        n_views=args.n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        device=device,
    )
    results.append(
        profile_variant(
            "baseline_eager",
            model_eager,
            x,
            K,
            R,
            t,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
    )

    # torch.compile with default options.
    if hasattr(torch, "compile"):
        try:
            model_compiled = torch.compile(model_eager, mode="default")
            results.append(
                profile_variant(
                    "torch_compile_default",
                    model_compiled,
                    x,
                    K,
                    R,
                    t,
                    n_warmup=max(args.n_warmup, 20),
                    n_iter=args.n_iter,
                )
            )
        except Exception as exc:  # pragma: no cover - compile may be unsupported
            print(f"torch.compile failed: {exc}")

    # Print comparison table.
    print("\nLatency comparison (ms):")
    print(f"{'variant':<25} {'mean':>8} {'std':>8} {'p50':>8} {'p99':>8}")
    for r in results:
        print(
            f"{r['variant']:<25} "
            f"{r['mean_ms']:>8.3f} {r['std_ms']:>8.3f} "
            f"{r['p50_ms']:>8.3f} {r['p99_ms']:>8.3f}"
        )

    baseline_ms = results[0]["mean_ms"]
    for r in results[1:]:
        speedup = baseline_ms / r["mean_ms"]
        r["speedup_vs_baseline"] = speedup
        print(f"{r['variant']} speedup vs baseline: {speedup:.2f}x")

    out = {
        "device": args.device,
        "batch_size": args.batch_size,
        "clip_len": args.clip_len,
        "n_views": args.n_views,
        "j": args.j,
        "d": args.d,
        "n_st_layers": args.n_st_layers,
        "residual_hidden": args.residual_hidden,
        "results": results,
    }

    json_out = args.json_out or f"outputs/profile_pp_optimizations_{args.device}.json"
    out_path = Path(json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
