#!/usr/bin/env python3
"""Profile model size and FLOPs for MotionFlow-MultiView variants.

This script is intentionally CPU-only and does not require a GPU, so it can be
used while the A800 GPUs are busy.  It reports parameter counts, model size,
and approximate forward-pass FLOPs.

Examples
--------
# v25 geometry fusion module
python scripts/analyze_model_size_flops.py \
    --model multiview_geometry_fusion_v25 \
    --batch-size 2 --t 4 --n-views 4 --joints 17

# Save a JSON report for later comparison
python scripts/analyze_model_size_flops.py \
    --model multiview_geometry_fusion_v25 \
    -o outputs/model_analysis_v25.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Make sure the repo root is on the path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.utils.model_analysis import (  # noqa: E402
    FlopsCounter,
    count_parameters,
    format_summary,
)
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (  # noqa: E402
    GeometryAwareCrossViewAttention,
    MultiViewGeometryFusionV25,
)


def _make_cameras(n_views: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a simple circular camera rig."""
    import numpy as np

    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    return (
        torch.from_numpy(np.stack(Ks)).float(),
        torch.from_numpy(np.stack(Rs)).float(),
        torch.from_numpy(np.stack(ts)).float(),
    )


def build_v25_inputs(
    batch_size: int = 2,
    t: int = 4,
    n_views: int = 4,
    n_joints: int = 17,
) -> Dict[str, Any]:
    """Build a plausible synthetic input tuple for ``MultiViewGeometryFusionV25``."""
    K0, R0, t0 = _make_cameras(n_views)
    temporal_len = t
    K = K0.unsqueeze(0).unsqueeze(0).expand(batch_size, temporal_len, -1, -1, -1).contiguous()
    R = R0.unsqueeze(0).unsqueeze(0).expand(batch_size, temporal_len, -1, -1, -1).contiguous()
    t = t0.unsqueeze(0).unsqueeze(0).expand(batch_size, temporal_len, -1, -1).contiguous()

    # Synthetic 2D points; we don't need geometric consistency for profiling.
    torch.manual_seed(42)
    points_2d = torch.randn(batch_size, temporal_len, n_views, n_joints, 2) * 0.1 + 256.0
    points_2d = torch.cat([points_2d, torch.ones(batch_size, temporal_len, n_views, n_joints, 1)], dim=-1)
    view_mask = torch.ones(batch_size, temporal_len, n_views).bool()
    return {
        "points_2d": points_2d,
        "K": K,
        "R": R,
        "t": t,
        "view_mask": view_mask,
    }


def _geometry_attention_flops(
    module: GeometryAwareCrossViewAttention,
    input: Tuple[Any, ...],
    output: torch.Tensor,
) -> int:
    """Custom FLOP handler for the geometry-aware cross-view attention block.

    Accounts for the attention matmuls that the generic hook-based counter
    cannot see because they are performed with ``torch.matmul`` rather than as
    separate ``nn.Linear`` modules.
    """
    tokens = input[0]  # (B, T, V, J, d)
    if tokens.dim() != 5:
        return 0
    B, T, V, J, d = tokens.shape
    n_heads = module.n_heads
    d_head = d // n_heads

    # QKV and output projection are already counted by the generic Linear hooks,
    # so we only add the attention matmuls here.
    # q @ k^T  : B*T*J * n_heads * V * V * d_head
    # attn @ v : B*T*J * n_heads * V * V * d_head
    flops = 2 * B * T * J * n_heads * V * V * d_head
    return flops


def analyze_v25(
    batch_size: int = 2,
    t: int = 4,
    n_views: int = 4,
    n_joints: int = 17,
    d: int = 128,
    n_heads: int = 4,
    n_geometry_layers: int = 2,
    n_ray_samples: int = 4,
) -> Dict[str, Any]:
    """Analyze the v25 geometry fusion module."""
    inputs = build_v25_inputs(batch_size=batch_size, t=t, n_views=n_views, n_joints=n_joints)
    model = MultiViewGeometryFusionV25(
        d=d,
        n_heads=n_heads,
        n_views=n_views,
        n_geometry_layers=n_geometry_layers,
        n_ray_samples=n_ray_samples,
        use_geometry_attention=True,
        use_learned_depth_triangulation=True,
        use_geometry_bundle_adjustment=False,
    )

    custom_handlers = {GeometryAwareCrossViewAttention: _geometry_attention_flops}
    counter = FlopsCounter(custom_handlers=custom_handlers)
    counter.register(model)
    try:
        with torch.no_grad():
            model(
                points_2d=inputs["points_2d"],
                K=inputs["K"],
                R=inputs["R"],
                t=inputs["t"],
                view_mask=inputs["view_mask"],
            )
    finally:
        counter.remove()

    params = count_parameters(model)
    summary = {
        "model": model.__class__.__name__,
        "config": {
            "d": d,
            "n_heads": n_heads,
            "n_views": n_views,
            "n_geometry_layers": n_geometry_layers,
            "n_ray_samples": n_ray_samples,
            "batch_size": batch_size,
            "t": t,
            "n_joints": n_joints,
        },
        "parameters": params.to_dict(),
        "flops": counter.flops_by_op.to_dict(),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile model size and FLOPs for MotionFlow-MultiView models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="multiview_geometry_fusion_v25",
        choices=["multiview_geometry_fusion_v25"],
        help="Model variant to profile.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for profiling.")
    parser.add_argument("--t", type=int, default=4, help="Temporal length for profiling.")
    parser.add_argument("--n-views", type=int, default=4, help="Number of camera views.")
    parser.add_argument("--joints", type=int, default=17, help="Number of joints.")
    parser.add_argument(
        "--d",
        type=int,
        default=128,
        help="Feature dimension for v25 ray tokens.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads for v25.",
    )
    parser.add_argument(
        "--n-geometry-layers",
        type=int,
        default=2,
        help="Number of geometry-attention layers for v25.",
    )
    parser.add_argument(
        "--n-ray-samples",
        type=int,
        default=4,
        help="Depth hypotheses per ray for v25.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional JSON file path to save the summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model == "multiview_geometry_fusion_v25":
        summary = analyze_v25(
            batch_size=args.batch_size,
            t=args.t,
            n_views=args.n_views,
            n_joints=args.joints,
            d=args.d,
            n_heads=args.n_heads,
            n_geometry_layers=args.n_geometry_layers,
            n_ray_samples=args.n_ray_samples,
        )
    else:
        raise NotImplementedError(f"Model variant '{args.model}' is not yet supported.")

    print(format_summary(summary))
    print()
    print("Config:", summary["config"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
