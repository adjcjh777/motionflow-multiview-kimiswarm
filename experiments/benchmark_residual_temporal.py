"""RTX 4090 inference benchmark for RayAttentionFusionModelTemporalResidual.

Measures end-to-end latency and throughput on CUDA and CPU for the residual
refinement temporal model. Synthetic 2D keypoints and a fixed 14-camera rig are
used for all batch sizes.

Usage (from repo root):
    D:/anaconda3/envs/mf/python.exe experiments/benchmark_residual_temporal.py

Outputs:
    outputs/benchmark_residual_temporal_4090.json
    outputs/benchmark_residual_temporal_4090.md
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)


class _FixedRig:
    """Single fixed 14-camera rig, built with pure torch/math."""

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


def _run_once(model, x, K, R, t):
    with torch.no_grad():
        _ = model(x, K=K, R=R, t=t)


def benchmark_device(
    model,
    device,
    batch_sizes,
    T,
    V,
    J,
    warmup,
    iters,
    rig,
):
    results = []
    model = model.to(device)
    model.eval()
    K, R, t = rig.to(device)

    for B in batch_sizes:
        x = _dummy_batch(B, T, V, J, device)

        with torch.no_grad():
            for _ in range(warmup):
                _run_once(model, x, K, R, t)
            if device.type == "cuda":
                torch.cuda.synchronize()

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

        total_frames = B * iters
        results.append(
            {
                "batch_size": B,
                "total_frames": total_frames,
                "total_seconds": elapsed,
                "latency_ms": elapsed / iters * 1000.0,
                "throughput_fps": total_frames / elapsed,
            }
        )
    return results


def _format_table(results, device_name):
    lines = [f"### {device_name}", ""]
    lines.append("| Batch | Latency (ms) | Throughput (fps) | Total frames |")
    lines.append("|-------|-------------:|-----------------:|-------------:|")
    for r in results:
        lines.append(
            f"| {r['batch_size']:>5} | {r['latency_ms']:>12.2f} | "
            f"{r['throughput_fps']:>16.2f} | {r['total_frames']:>12} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--n_views", type=int, default=14)
    parser.add_argument("--j", type=int, default=28)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

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

    # CPU: only B=1 to keep runtime reasonable
    cpu_device = torch.device("cpu")
    all_results["devices"]["cpu"] = benchmark_device(
        model, cpu_device, [1], T, V, J, args.warmup, args.iters, rig
    )

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda:0")
        all_results["devices"]["gpu"] = benchmark_device(
            model, gpu_device, args.batch_sizes, T, V, J, args.warmup, args.iters, rig
        )
    else:
        print("CUDA not available; skipping GPU benchmark.")

    # Save JSON
    json_path = out_dir / "benchmark_residual_temporal_4090.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    # Save Markdown
    md_lines = [
        "# RTX 4090 Residual Temporal Model Benchmark",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        f"| Model | `{all_results['model']}` |",
        f"| Params | {param_count:,} |",
        f"| d | {d} |",
        f"| residual_hidden | {residual_hidden} |",
        f"| Views (V) | {V} |",
        f"| Joints (J) | {J} |",
        f"| Clip length (T) | {T} |",
        f"| Warmup | {args.warmup} |",
        f"| Iters | {args.iters} |",
        "",
        _format_table(all_results["devices"]["cpu"], "CPU (B=1)"),
    ]

    if "gpu" in all_results["devices"]:
        md_lines.append(_format_table(all_results["devices"]["gpu"], "GPU (cuda:0)"))

    md_path = out_dir / "benchmark_residual_temporal_4090.md"
    md_path.write_text("\n".join(md_lines))

    print(f"Saved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
