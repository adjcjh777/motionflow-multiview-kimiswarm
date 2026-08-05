#!/usr/bin/env python3
"""A800 inference benchmark for RayAttentionFusionModelTemporalResidual.

Mirrors ``experiments/benchmark_residual_temporal.py`` but targets the
NVIDIA A800 (80 GB) with larger batch sizes and peak memory reporting.

Usage (from repo root, bare metal or inside Docker):
    python scripts/benchmark_a800.py --batch_sizes 1 8 16 32 64 --iters 200

Outputs:
    outputs/benchmark_a800/benchmark_a800.json
    outputs/benchmark_a800/benchmark_a800.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch

# Make repository modules importable when script is run from repo root.
ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)


def _set_deterministic(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _FixedRig:
    """Single fixed multi-camera rig, built with pure torch/math."""

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

    def to(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.K.to(device), self.R.to(device), self.t.to(device)


def _dummy_batch(batch_size: int, T: int, V: int, J: int, device: torch.device) -> torch.Tensor:
    x = torch.randn(batch_size, T, V, J, 3, device=device)
    x[..., 2] = torch.sigmoid(x[..., 2])
    return x


def _run_once(model, x, K, R, t) -> None:
    with torch.no_grad():
        _ = model(x, K=K, R=R, t=t)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024.0**2


def benchmark_device(
    model: torch.nn.Module,
    device: torch.device,
    batch_sizes: List[int],
    T: int,
    V: int,
    J: int,
    warmup: int,
    iters: int,
    rig: _FixedRig,
) -> List[dict]:
    results: List[dict] = []
    model = model.to(device)
    model.eval()
    K, R, t = rig.to(device)

    for B in batch_sizes:
        x = _dummy_batch(B, T, V, J, device)

        _reset_peak_memory(device)

        with torch.no_grad():
            for _ in range(warmup):
                _run_once(model, x, K, R, t)
            if device.type == "cuda":
                torch.cuda.synchronize()

        _reset_peak_memory(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iters):
                _run_once(model, x, K, R, t)
            if device.type == "cuda":
                torch.cuda.synchronize()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_mb = _peak_memory_mb(device)

        total_frames = B * iters
        results.append(
            {
                "batch_size": B,
                "total_frames": total_frames,
                "total_seconds": elapsed,
                "latency_ms": elapsed / iters * 1000.0,
                "throughput_fps": total_frames / elapsed,
                "peak_memory_mb": peak_mb,
            }
        )
    return results


def _format_table(results: List[dict], device_name: str) -> str:
    lines = [f"### {device_name}", ""]
    lines.append(
        "| Batch | Latency (ms) | Throughput (fps) | Peak memory (MB) | Total frames |"
    )
    lines.append(
        "|-------|-------------:|-----------------:|-----------------:|-------------:|"
    )
    for r in results:
        lines.append(
            f"| {r['batch_size']:>5} | {r['latency_ms']:>12.2f} | "
            f"{r['throughput_fps']:>16.2f} | {r['peak_memory_mb']:>16.2f} | "
            f"{r['total_frames']:>12} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64],
        help="Batch sizes to benchmark (A800 has 80 GB).",
    )
    parser.add_argument("--n_views", type=int, default=14)
    parser.add_argument("--j", type=int, default=28)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/benchmark_a800")
    args = parser.parse_args()

    _set_deterministic(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    V = args.n_views
    J = args.j
    T = args.clip_len
    d = args.d
    residual_hidden = args.residual_hidden

    model = RayAttentionFusionModelTemporalResidual(
        j=J,
        d=d,
        n_views=V,
        n_heads=4,
        n_joint_layers=1,
        n_temporal_layers=2,
        max_temporal_len=256,
        residual_hidden=residual_hidden,
    )

    param_count = sum(p.numel() for p in model.parameters())

    all_results = {
        "model": "RayAttentionFusionModelTemporalResidual",
        "params": param_count,
        "d": d,
        "residual_hidden": residual_hidden,
        "views": V,
        "joints": J,
        "clip_len": T,
        "warmup": args.warmup,
        "iters": args.iters,
        "devices": {},
    }

    rig = _FixedRig(V, torch.device("cpu"))

    # CPU: only B=1 to keep runtime reasonable.
    cpu_device = torch.device("cpu")
    all_results["devices"]["cpu"] = benchmark_device(
        model, cpu_device, [1], T, V, J, args.warmup, args.iters, rig
    )

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(gpu_device) or "cuda:0"
        all_results["devices"]["gpu"] = benchmark_device(
            model,
            gpu_device,
            args.batch_sizes,
            T,
            V,
            J,
            args.warmup,
            args.iters,
            rig,
        )
        all_results["gpu_name"] = gpu_name
    else:
        print("CUDA not available; skipping GPU benchmark.")

    # Save JSON
    json_path = out_dir / "benchmark_a800.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    # Save Markdown
    md_lines = [
        "# A800 Residual Temporal Model Benchmark",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        f"| Model | `{all_results['model']}` |",
        f"| Params | {param_count:,} |",
        f"| d | {d} |",
        f"| residual_hidden | {residual_hidden} |",
        f"| GPU | {all_results.get('gpu_name', 'N/A')} |",
        f"| Views (V) | {V} |",
        f"| Joints (J) | {J} |",
        f"| Clip length (T) | {T} |",
        f"| Warmup | {args.warmup} |",
        f"| Iters | {args.iters} |",
        "",
        _format_table(all_results["devices"]["cpu"], "CPU (B=1)"),
    ]

    if "gpu" in all_results["devices"]:
        md_lines.append(_format_table(all_results["devices"]["gpu"], f"GPU ({gpu_name})"))

    md_path = out_dir / "benchmark_a800.md"
    md_path.write_text("\n".join(md_lines))

    print(f"Saved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
