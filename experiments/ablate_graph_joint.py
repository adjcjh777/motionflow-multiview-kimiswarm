"""CPU smoke ablation for the skeleton-graph joint-relation module.

Runs the ``GraphJointRelation`` block on synthetic data and reports per-layer
output statistics for several skeleton-graph variants (bone-only, bone+
symmetry, bone+symmetry+cross-view).  No GPU or training is performed.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/ablate_graph_joint.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow ``torch`` on Windows/WSL builds that trip over duplicate OpenMP libs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.models.graph_joint_relation import (
    GraphJointRelation,
    build_edge_index,
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)


def set_seed(seed: int = 2027):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_edge_subset(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    keep_bone: bool = True,
    keep_sym: bool = True,
    keep_cross: bool = True,
):
    """Return a masked edge list/edge-type pair for ablation studies."""
    mask = torch.zeros(edge_type.size(0), dtype=torch.bool)
    if keep_bone:
        mask = mask | (edge_type == 0)
    if keep_sym:
        mask = mask | (edge_type == 1)
    if keep_cross:
        mask = mask | (edge_type == 2)
    return edge_index[:, mask], edge_type[mask]


def run_variant(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    module: GraphJointRelation,
    variant_name: str,
):
    """Run one graph variant and return a structured report."""
    x = x.detach().clone().requires_grad_(True)
    out, intermediates = module(x, edge_index, edge_type, return_intermediates=True)

    # Use a non-degenerate scalar objective so gradients are numerically visible.
    loss = (out ** 2).mean()
    loss.backward()

    layer_stats = []
    for i, layer_out in enumerate(intermediates):
        layer_stats.append(
            {
                "mean": float(layer_out.mean().detach()),
                "std": float(layer_out.std().detach()),
                "min": float(layer_out.min().detach()),
                "max": float(layer_out.max().detach()),
                "shape": tuple(layer_out.shape),
            }
        )

    return {
        "variant": variant_name,
        "edges": edge_index.size(1),
        "edge_type_counts": {
            "bone": int((edge_type == 0).sum()),
            "symmetry": int((edge_type == 1).sum()),
            "cross_view": int((edge_type == 2).sum()),
        },
        "output_shape": tuple(out.shape),
        "output_mean": float(out.mean().detach()),
        "output_std": float(out.std().detach()),
        "grad_norm": float(x.grad.norm()),
        "layer_stats": layer_stats,
    }


def run_skeleton_ablation(parents, symmetry_pairs, name: str, n_views: int, d: int, num_layers: int, num_heads: int):
    """Run all graph variants for a single skeleton preset."""
    print(f"\n=== Skeleton: {name} (J={len(parents)}, V={n_views}, d={d}) ===")

    edge_index, edge_type = build_edge_index(parents, symmetry_pairs, n_views=n_views, j=len(parents))
    module = GraphJointRelation(in_dim=d, n_views=n_views, num_layers=num_layers, num_heads=num_heads)

    B = 2
    x = torch.randn(B, n_views, len(parents), d)

    configs = [
        ("bone only", True, False, False),
        ("bone + symmetry", True, True, False),
        ("bone + symmetry + cross-view", True, True, True),
    ]

    results = []
    for variant_name, keep_bone, keep_sym, keep_cross in configs:
        ei, et = make_edge_subset(edge_index, edge_type, keep_bone, keep_sym, keep_cross)
        if ei.numel() == 0:
            continue
        report = run_variant(x, ei, et, module, variant_name)
        results.append(report)

        print(f"\n-- {variant_name} --")
        print(f"  directed edges: {report['edges']}")
        print(f"  edge counts: {report['edge_type_counts']}")
        print(f"  output shape: {report['output_shape']}")
        print(f"  output mean/std: {report['output_mean']:.4f} / {report['output_std']:.4f}")
        print(f"  grad norm: {report['grad_norm']:.2e}")
        for i, stats in enumerate(report["layer_stats"]):
            print(
                f"  layer {i}: shape={stats['shape']}, "
                f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="CPU smoke ablation for GraphJointRelation")
    parser.add_argument("--d", type=int, default=64, help="Feature dimension (must be divisible by num_heads)")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of graph attention layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_views", type=int, default=4, help="Number of views")
    parser.add_argument("--seed", type=int, default=2027, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print("GraphJointRelation CPU smoke ablation")
    print(f"Config: d={args.d}, num_layers={args.num_layers}, num_heads={args.num_heads}, n_views={args.n_views}")

    run_skeleton_ablation(
        H36M_17_PARENTS,
        H36M_17_SYMMETRY_PAIRS,
        "H36M-17",
        args.n_views,
        args.d,
        args.num_layers,
        args.num_heads,
    )
    run_skeleton_ablation(
        MPI_INF_3DHP_28_PARENTS,
        MPI_INF_3DHP_28_SYMMETRY_PAIRS,
        "MPI-INF-3DHP-28",
        args.n_views,
        args.d,
        args.num_layers,
        args.num_heads,
    )

    print("\nAll CPU smoke checks passed.")


if __name__ == "__main__":
    main()
