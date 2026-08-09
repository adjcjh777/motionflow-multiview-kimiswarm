#!/usr/bin/env python3
"""Benchmark a checkpoint with the standard MPJPE@k protocol.

Examples
--------
Single dataset:
    python scripts/run_mpjpe_at_k_benchmark.py \\
        --checkpoint outputs/omniview_fusion_v46_svg.pth \\
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \\
        --k_values 2 3 4 14 \\
        --output_dir outputs/mpjpe_at_k

Dataset manifest:
    python scripts/run_mpjpe_at_k_benchmark.py \\
        --checkpoint outputs/omniview_fusion_v46_svg.pth \\
        --dataset_manifest configs/splits/v48_eval_manifest.txt \\
        --k_values 2 3 4 14 \\
        --output_dir outputs/mpjpe_at_k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.eval_variable_views import (
    _build_omniview_v5_model,
    _load_config,
    _load_dataset_manifest,
    _load_npz_dataset,
    _resolve_config_path,
)
from motionflow_mv.eval.mpjpe_at_k_protocol import (
    evaluate_mpjpe_at_k,
    print_mpjpe_at_k_table,
    write_mpjpe_at_k_csv,
    write_mpjpe_at_k_json,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MPJPE@k benchmark for a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--model_class", type=str, default="omniview_v5", help="Model class to instantiate.")
    parser.add_argument("--config", type=str, default=None, help="Optional training config JSON/YAML.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a single .npz dataset.")
    parser.add_argument("--dataset_manifest", type=str, default=None, help="Path to a manifest of datasets.")
    parser.add_argument("--k_values", type=int, nargs="+", default=[2, 3, 4, 8], help="View counts to evaluate.")
    parser.add_argument("--num_subsets_per_k", type=int, default=None, help="Subsets per k; None enumerates all.")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for subset sampling.")
    parser.add_argument("--align", type=str, default="none", choices=["none", "pa", "root"], help="Alignment mode.")
    parser.add_argument("--hardened", action="store_true", help="Use hardened variable-view wrapper.")
    parser.add_argument("--output_dir", type=str, default="outputs/mpjpe_at_k", help="Output directory.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; inferred if not set.")
    return parser


def _load_datasets(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.dataset:
        return [(Path(args.dataset).stem, args.dataset)]
    if args.dataset_manifest:
        return _load_dataset_manifest(args.dataset_manifest)
    raise ValueError("Either --dataset or --dataset_manifest must be provided.")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    if args.model_class != "omniview_v5":
        raise NotImplementedError("Only model_class='omniview_v5' is currently supported.")

    config_path = _resolve_config_path(args, args.checkpoint)
    config = _load_config(config_path) if config_path else {}

    # n_joints and n_views are inferred from the first dataset.
    first_npz = _load_datasets(args)[0][1]
    _, confidences, _, _ = _load_npz_dataset(first_npz)
    n_joints = confidences.shape[-1]
    n_views = confidences.shape[-2]

    model = _build_omniview_v5_model(config, args.checkpoint, n_joints, n_views, device)

    datasets = _load_datasets(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[int, Dict]] = {}
    for name, npz_path in datasets:
        print(f"\nEvaluating dataset: {name}")
        points_2d, confidences, joints_3d, cameras = _load_npz_dataset(npz_path)
        results = evaluate_mpjpe_at_k(
            model,
            points_2d,
            confidences,
            joints_3d,
            cameras,
            k_values=args.k_values,
            clip_len=args.clip_len,
            num_subsets_per_k=args.num_subsets_per_k,
            seed=args.seed,
            align=args.align,
            device=device,
            hardened=args.hardened,
        )
        all_results[name] = results
        print_mpjpe_at_k_table(results, label=f"{name} MPJPE@k")
        write_mpjpe_at_k_csv(results, output_dir / f"{name}_mpjpe_at_k.csv")
        write_mpjpe_at_k_json(results, output_dir / f"{name}_mpjpe_at_k.json")

    # Cross-dataset summary if more than one dataset.
    if len(all_results) > 1:
        summary_path = output_dir / "mpjpe_at_k_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved cross-dataset summary to {summary_path}")


if __name__ == "__main__":
    main()
