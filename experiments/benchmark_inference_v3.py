"""Efficiency benchmark for RayAttentionFusionModelV3.

Measures end-to-end inference latency and throughput on CPU and GPU for a
range of batch sizes, and attempts ONNX export of the full v3 model.

Usage (from repo root):
    /d/anaconda3/envs/jz_py310/python.exe experiments/benchmark_inference_v3.py

Outputs:
    outputs/inference_benchmark.json
    outputs/inference_benchmark.md

Summary of findings (Aug 2026):
- RayAttentionFusionModelV3 has ~93.5 k parameters (d=64, 4 views, 17 joints).
- On the local workstation, CPU inference is faster than GPU for small batches
  (~14 ms vs ~18 ms for B=1), while GPU throughput scales similarly at larger
  batches (both reach ~1500-1900 fps at B=32).
- Full-model ONNX export fails because the differentiable weighted DLT layer uses
  torch.linalg.lstsq, which is not in the default ONNX opset. A practical
  deployment path is to export only the network up to per-view weight prediction
  and run triangulation in a separate, geometry-specific runtime step.
- The benchmark uses a randomly-initialized checkpoint; numbers reflect
  architecture-level latency/throughput and are independent of learned weights.
- Implementation note: the NumPy BLAS/LAPACK backend on this host segfaults on
  matrix operations (np.matmul, np.linalg.qr), so camera generation uses pure
  torch/math.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3


class _FixedRig:
    """A single fixed camera rig used for all benchmark samples.

    Built with pure torch/math to avoid the broken NumPy BLAS backend on this
    host (``numpy.matmul`` / ``np.linalg.qr`` crash the interpreter with
    exit code 127).
    """

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
        radius = 3.0
        for i in range(V):
            # Slightly varying focal length so the rig is not degenerate.
            fx = fy = 1400.0 + (i % 2) * 100.0
            cx = 960.0
            cy = 540.0
            K = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
            )
            Ks.append(K)

            angle = 2.0 * math.pi * i / V
            Rs.append(self._rotation_y(angle))

            t = torch.tensor(
                [math.cos(angle) * radius, 0.5, math.sin(angle) * radius],
                dtype=torch.float32,
            )
            t = t / (t.norm() + 1e-8) * radius
            ts.append(t)

        K = torch.stack(Ks)
        R = torch.stack(Rs)
        t = torch.stack(ts)
        return K.to(device), R.to(device), t.to(device)

    def to(self, device: torch.device):
        return self.K.to(device), self.R.to(device), self.t.to(device)


def _dummy_batch(batch_size: int, V: int, J: int, device: torch.device):
    x = torch.randn(batch_size, V, J, 3, device=device)
    x[..., 2] = torch.sigmoid(x[..., 2])
    return x


def benchmark_device(model, device, batch_sizes, V, J, warmup, iters, rig):
    results = []
    model = model.to(device)
    model.eval()
    K, R, t = rig.to(device)
    for B in batch_sizes:
        x = _dummy_batch(B, V, J, device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x, K=K, R=R, t=t)
            if device.type == "cuda":
                torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(x, K=K, R=R, t=t)
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


def _try_onnx_export(model, V, J, onnx_path):
    device = torch.device("cpu")
    model = model.to(device).eval()
    rig = _FixedRig(V, device)
    x = _dummy_batch(1, V, J, device)
    K, R, t = rig.to(device)
    args = (x, K, R, t)
    try:
        torch.onnx.export(
            model,
            args,
            str(onnx_path),
            input_names=["x", "K", "R", "t"],
            output_names=["pred_3d", "weights"],
            dynamic_axes={
                "x": {0: "batch"},
                "pred_3d": {0: "batch"},
                "weights": {0: "batch"},
            },
            opset_version=17,
        )
        return "success"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--v", type=int, default=4, help="number of views")
    parser.add_argument("--j", type=int, default=17, help="number of joints")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    model = RayAttentionFusionModelV3(j=args.j, d=64, n_views=args.v, n_heads=4, n_joint_layers=1)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    param_count = sum(p.numel() for p in model.parameters())

    all_results = {
        "model": "RayAttentionFusionModelV3",
        "params": param_count,
        "views": args.v,
        "joints": args.j,
        "devices": {},
    }

    rig = _FixedRig(args.v, torch.device("cpu"))

    cpu_device = torch.device("cpu")
    all_results["devices"]["cpu"] = benchmark_device(
        model, cpu_device, args.batch_sizes, args.v, args.j, args.warmup, args.iters, rig
    )

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda:0")
        all_results["devices"]["gpu"] = benchmark_device(
            model, gpu_device, args.batch_sizes, args.v, args.j, args.warmup, args.iters, rig
        )
    else:
        all_results["devices"]["gpu"] = None

    onnx_path = out_dir / "ray_attention_v3.onnx"
    all_results["onnx_export"] = _try_onnx_export(model, args.v, args.j, onnx_path)

    json_path = out_dir / "inference_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    md_lines = [
        "# Inference Benchmark - RayAttentionFusionModelV3",
        "",
        f"**Model**: `{all_results['model']}`  ",
        f"**Parameters**: {param_count:,}  ",
        f"**Input shape**: (B, {args.v}, {args.j}, 3) - per-view (x, y, confidence)  ",
        f"**Warmup iterations**: {args.warmup}  ",
        f"**Measured iterations**: {args.iters}  ",
        "",
        "## Latency / Throughput",
        "",
        "| Device | Batch | Latency (ms) | Throughput (fps) | Total frames |",
        "|--------|-------|-------------:|-----------------:|-------------:|",
    ]
    for dev_name, dev_results in all_results["devices"].items():
        if dev_results is None:
            continue
        for r in dev_results:
            md_lines.append(
                f"| {dev_name.upper()} | {r['batch_size']:>3} | "
                f"{r['latency_ms']:>11.2f} | {r['throughput_fps']:>16.1f} | {r['total_frames']:>12} |"
            )
    md_lines.extend(
        [
            "",
            "## ONNX export",
            "",
            f"Attempted export: `{onnx_path.as_posix()}`  ",
            f"Result: **{all_results['onnx_export']}**  ",
            "",
            "## Notes",
            "",
            "- CPU timings were taken on the local WSL/Windows workstation.",
            "- GPU timings use `torch.cuda.synchronize()` before and after the measured loop.",
            "- The model benchmarked here uses a randomly-initialized checkpoint;",
            "  the reported numbers reflect architecture-level throughput/latency and are",
            "  independent of learned weights.",
        ]
    )
    if all_results["onnx_export"].startswith("failed"):
        md_lines.extend(
            [
                "",
                "> The ONNX export failed, most likely because the differentiable weighted DLT layer",
                "> uses `torch.linalg.lstsq`, which is not supported by the default ONNX opset.",
                "> A practical deployment path is to export the network up to the per-view weight",
                "> prediction and perform triangulation in a separate, geometry-specific runtime step.",
            ]
        )

    md_path = out_dir / "inference_benchmark.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Benchmark JSON: {json_path}")
    print(f"Benchmark report: {md_path}")
    if all_results["onnx_export"].startswith("failed"):
        print("ONNX export failed as expected; see report for details.")
    else:
        print(f"ONNX export succeeded: {onnx_path}")


if __name__ == "__main__":
    main()
