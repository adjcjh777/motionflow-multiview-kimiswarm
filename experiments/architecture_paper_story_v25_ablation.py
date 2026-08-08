"""Paper-story ablation smoke test for the v25 architecture.

Instantiates ``OmniMultiViewFusionV5`` with the v25 geometry-fusion stack and
ablates the three headline v25 modules:

* geometry-aware cross-view attention
* learned depth-proposal triangulation
* geometry bundle adjustment (GeoBA)

Each variant is run on synthetic calibrated data and a small reproducible
ablation table + bar chart are written to ``outputs/architecture_paper_story/``.

Intended as a smoke test on the local RTX 4090 (or CPU fallback) that can be
re-run whenever the v25 architecture is modified.

Usage
-----
    python experiments/architecture_paper_story_v25_ablation.py \
        --config configs/architecture_paper_story_v25_smoke.yaml \
        --output_dir outputs/architecture_paper_story
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _make_synthetic_data(cameras, *, B: int, T: int, J: int, device: torch.device, noise_std: float = 1.0):
    """Return a synthetic (input, target_3d) pair in world coordinates.

    The 3D target is projected through the provided cameras, Gaussian noise is
    added in pixels, and confidences are set to 1.
    """
    V = len(cameras)
    # Random 3D joints in a ~1 m cube around the origin.
    target_3d = (torch.rand(B, T, J, 3, device=device) - 0.5) * 1.0
    Xh = torch.cat([target_3d, torch.ones(B, T, J, 1, device=device)], dim=-1)

    points_2d = []
    for cam in cameras:
        P = torch.from_numpy(cam.projection_matrix).float().to(device)  # (3, 4)
        x_hom = Xh @ P.T  # (B, T, J, 3)
        x_hom = x_hom[..., :2] / (x_hom[..., 2:3] + 1e-6)
        x_hom = x_hom + torch.randn_like(x_hom) * noise_std
        points_2d.append(x_hom)

    points_2d = torch.stack(points_2d, dim=2)  # (B, T, V, J, 2)
    confidences = torch.ones(B, T, V, J, device=device)
    x = torch.cat([points_2d, confidences.unsqueeze(-1)], dim=-1)  # (B, T, V, J, 3)
    return x, target_3d


def _build_model_kwargs(base: dict, variant: dict) -> dict:
    kwargs = dict(base)
    for k, v in variant.items():
        if k == "name":
            continue
        kwargs[k] = v
    return kwargs


def _mpjpe_mm(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return MPJPE in millimetres (assuming coordinates are in metres)."""
    return (pred - target).norm(dim=-1).mean().item() * 1000.0


def _measure_forward(model, x, cameras, device, n_iters: int = 3):
    """Measure average forward latency in milliseconds."""
    # Warm-up
    for _ in range(2):
        with torch.no_grad():
            _ = model(x, cameras=cameras)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = model(x, cameras=cameras)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_iters
    else:
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = model(x, cameras=cameras)
        ms = (time.perf_counter() - t0) * 1000.0 / n_iters
    return ms


def _generate_bar_chart(results, output_path: Path, metric_key: str, title: str, ylabel: str):
    names = [r["name"] for r in results]
    values = [r[metric_key] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color="steelblue")
    bars[0].set_color("darkorange")  # highlight full model
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_table(results, output_path: Path):
    lines = [
        "# v25 Architecture Component Ablation (smoke test)\n\n",
        "| Variant | geometry_attention | learned_depth | GeoBA | Params | Forward (ms) | ",
        "Δ to full (mm) | Proxy MPJPE (mm) |\n",
        "|---|---|---|---|---:|---:|---:|---:|\n",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['geometry_attention']} | {r['learned_depth']} | {r['geoba']} | "
            f"{r['params']:,} | {r['forward_ms']:.2f} | {r['delta_mm']:.2f} | {r['proxy_mpjpe_mm']:.2f} |\n"
        )
    lines.append(
        "\n**Note:** The v25 geometry-fusion modules are intentionally identity at "
        "initialization (warm-startable).  Consequently the prediction delta to "
        "the full model is near zero before training.  This harness is meant as a "
        "reproducible scaffold; re-run it after training for meaningful ablation "
        "numbers.\n"
    )
    output_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="v25 architecture ablation smoke test")
    parser.add_argument("--config", type=str, default="configs/architecture_paper_story_v25_smoke.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/architecture_paper_story")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--J", type=int, default=17)
    parser.add_argument("--noise_std", type=float, default=1.0)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = cfg["base"]
    n_views = base["n_views"]

    # Imports are local so the script fails fast if dependencies are missing.
    from motionflow_mv.fusion.omniview_fusion_v5 import (  # noqa: E402
        OmniMultiViewFusionV5,
        _make_cameras,
    )

    cameras = _make_cameras(n_views)
    x, target_3d = _make_synthetic_data(
        cameras, B=args.B, T=args.T, J=args.J, device=device, noise_std=args.noise_std
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    full_pred = None

    for variant in cfg["variants"]:
        kwargs = _build_model_kwargs(base, variant)
        name = variant.get("name", "unnamed")

        torch.manual_seed(42)
        model = OmniMultiViewFusionV5(**kwargs).to(device)
        model.eval()

        with torch.no_grad():
            out = model(x, cameras=cameras)
        pred = out[0]

        if full_pred is None:
            full_pred = pred
            delta_mm = 0.0
        else:
            delta_mm = _mpjpe_mm(pred, full_pred)

        params = sum(p.numel() for p in model.parameters())
        forward_ms = _measure_forward(model, x, cameras, device, n_iters=3)
        proxy_mpjpe_mm = _mpjpe_mm(pred, target_3d)

        results.append(
            {
                "name": name,
                "geometry_attention": kwargs.get("v25_use_geometry_attention", False),
                "learned_depth": kwargs.get("v25_use_learned_depth_triangulation", False),
                "geoba": kwargs.get("v25_use_geometry_bundle_adjustment", False),
                "params": params,
                "forward_ms": forward_ms,
                "delta_mm": delta_mm,
                "proxy_mpjpe_mm": proxy_mpjpe_mm,
            }
        )

    _generate_bar_chart(
        results,
        output_dir / "v25_ablation_delta.png",
        metric_key="delta_mm",
        title="v25 architecture ablation: prediction delta to full model",
        ylabel="Δ MPJPE to full model (mm)",
    )
    _generate_bar_chart(
        results,
        output_dir / "v25_ablation_proxy_mpjpe.png",
        metric_key="proxy_mpjpe_mm",
        title="v25 architecture ablation: proxy MPJPE (random weights)",
        ylabel="Proxy MPJPE (mm)",
    )
    _generate_bar_chart(
        results,
        output_dir / "v25_ablation_params.png",
        metric_key="params",
        title="v25 architecture ablation: parameter count",
        ylabel="Number of parameters",
    )

    md_path = output_dir / "v25_ablation_table.md"
    _write_markdown_table(results, md_path)

    print(f"Device: {device}")
    print(f"Results written to: {output_dir}")
    print(f"Markdown table: {md_path}")
    for r in results:
        print(
            f"  {r['name']:<25} params={r['params']:,}  fwd={r['forward_ms']:.2f} ms  "
            f"delta={r['delta_mm']:.2f} mm  proxy={r['proxy_mpjpe_mm']:.2f} mm"
        )


if __name__ == "__main__":
    main()
