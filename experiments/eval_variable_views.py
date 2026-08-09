"""Benchmark variable view-count inference for fixed-view fusion models.

Usage (synthetic smoke test, no data/checkpoint required):
    python experiments/eval_variable_views.py --n_views 6 --j 17 --clip_len 9

Usage with a real dataset and checkpoint:
    python experiments/eval_variable_views.py \\
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \\
        --checkpoint outputs/ray_attention_temporal_residual_v2.pth \\
        --clip_len 13 --d 64

Usage with OmniMultiViewFusionV5 (v46/v47 comparison):
    python experiments/eval_variable_views.py \\
        --model_class omniview_v5 \\
        --checkpoint outputs/v47_temporal_svg.pth \\
        --config outputs/v47_temporal_svg.config.json \\
        --compare_v46_v47 \\
        --dataset data/webbridge/... --output_csv v46_v47.csv

Usage with a dataset manifest for per-dataset reporting:
    python experiments/eval_variable_views.py \\
        --model_class omniview_v5 \\
        --checkpoint outputs/v47_temporal_svg.pth \\
        --config outputs/v47_temporal_svg.config.json \\
        --dataset_manifest docs/swarm_iter25/v48_eval_manifest.txt \\
        --output_csv v48_per_dataset.csv
"""

import argparse
import inspect
import json
import math
import sys
from argparse import Namespace
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import mpjpe as mpjpe_metric
from motionflow_mv.eval.mpjpe_at_k_protocol import (
    compute_mpjpe_at_k,
    evaluate_mpjpe_at_k,
    generate_view_subsets,
    print_mpjpe_at_k_table,
    temporal_jerk,
    write_mpjpe_at_k_csv,
    write_mpjpe_at_k_json,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
)
from motionflow_mv.fusion.variable_view_inference import (
    HardenedVariableViewInferenceWrapper,
    VariableViewInferenceWrapper,
    generate_view_subsets,
    prepare_variable_view_input,
)


def _temporal_jerk(poses_m: np.ndarray) -> float:
    """Mean magnitude of the 3rd temporal derivative of ``poses_m`` (mm)."""
    if poses_m.shape[0] < 4:
        return 0.0
    # Third finite difference along the time axis.
    third_diff = np.diff(poses_m, n=3, axis=0)  # (T-3, J, 3) in metres
    jerk_mm = np.linalg.norm(third_diff, axis=-1).mean() * 1000.0
    return float(jerk_mm)


def _load_config(path: str) -> Dict[str, Any]:
    """Load a flat JSON/YAML configuration saved by the trainer."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix in {".yaml", ".yml"}:
        import yaml
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(p, "r") as f:
            cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a top-level dictionary")
    return cfg


def _load_npz_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load a WebBridge-style .npz and return (points_2d, confidences, joints_3d, cameras)."""
    data = np.load(path)
    points_2d = data["points_2d"]
    confidences = data["confidences"]
    joints_3d = data["joints_3d"]
    n_views = data["camera_K"].shape[0]
    from motionflow_mv.calibration.camera import Camera

    cameras = []
    for v in range(n_views):
        cameras.append(
            Camera(
                K=data["camera_K"][v],
                R=data["camera_R"][v],
                t=data["camera_t"][v],
            )
        )
    return points_2d, confidences, joints_3d, cameras


def _load_dataset_manifest(path: str) -> List[Tuple[str, str]]:
    """Parse a manifest file into (dataset_name, npz_path) pairs.

    Each non-empty, non-comment line must contain a dataset name and a path
    separated by whitespace.  Lines starting with ``#`` are ignored.
    """
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            entries.append((parts[0], parts[1]))
    return entries


def _build_omniview_v5_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    n_joints: int,
    n_views: int,
    device: torch.device,
) -> torch.nn.Module:
    """Build ``OmniMultiViewFusionV5`` from saved config and checkpoint."""
    from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5

    sig = inspect.signature(OmniMultiViewFusionV5.__init__)
    param_names = set(sig.parameters.keys())

    # The trainer exposes ``--v47_use_view_count_conditioning`` but the model
    # constructor uses a more explicit parameter name.
    key_map = {
        "v47_use_view_count_conditioning": "v47_temporal_use_view_count_conditioning",
    }

    kwargs: Dict[str, Any] = {"j": n_joints, "n_views": n_views}
    for k, v in config.items():
        mk = key_map.get(k, k)
        if mk in param_names and mk not in kwargs:
            kwargs[mk] = v

    # Provide sensible defaults for v47 parameters that may be absent from an
    # older config.
    defaults = {
        "use_v47_temporal_aggregation": False,
        "v47_temporal_d_model": 64,
        "v47_temporal_n_heads": 4,
        "v47_temporal_num_layers": 2,
        "v47_temporal_window": None,
        "v47_temporal_dropout": 0.1,
        "v47_temporal_residual_gate_init": 0.0,
        "v47_temporal_use_view_count_conditioning": True,
    }
    for dk, dv in defaults.items():
        kwargs.setdefault(dk, dv)

    model = OmniMultiViewFusionV5(**kwargs)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    # Rebuild the (view, joint) graph for the target skeleton.  The graph edge
    # buffers are saved in the checkpoint, so loading a model trained on a
    # different skeleton (e.g. 28-joint MPI-INF-3DHP) and evaluating on another
    # (e.g. 17-joint H36M) would otherwise trigger out-of-bounds gathers.
    if hasattr(model, "rebuild_graph"):
        dataset = "mpiinf3dhp" if n_joints == 28 else "h36m"
        model.rebuild_graph(n_joints, dataset=dataset)

    model.to(device)
    model.eval()
    return model


def _resolve_config_path(args: argparse.Namespace, checkpoint_path: Optional[str]) -> Optional[str]:
    """Return the path to the model config, or ``None``."""
    if args.config is not None:
        return args.config
    if checkpoint_path is None:
        return None
    candidate = Path(checkpoint_path).with_suffix(".config.json")
    return str(candidate) if candidate.exists() else None


def _make_synthetic_cameras(n_views: int = 4):
    """Build the same circular rig used by the ray-attention smoke tests."""
    from motionflow_mv.calibration.camera import Camera
    cameras = []
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


def _make_synthetic_dataset(n_views: int, j: int, n_frames: int):
    """Return a smoke-test dataset.

    The 2D observations are random values in [0, 1] and the 3D ground truth
    is a dummy skeleton.  This is intentionally geometrically meaningless; it
    only exercises the variable-view inference and benchmark logic on CPU.
    For a real accuracy benchmark, supply a calibrated dataset with --dataset.
    """
    rng = np.random.default_rng(2024)
    cameras = _make_synthetic_cameras(n_views)
    points_2d = rng.uniform(0, 1, size=(n_frames, n_views, j, 2)).astype(np.float32)
    confidences = np.ones((n_frames, n_views, j), dtype=np.float32)
    joints_3d = rng.uniform(-1, 1, size=(n_frames, j, 3)).astype(np.float32)
    return points_2d, confidences, joints_3d, cameras


MODEL_CLASSES = {
    "temporal_residual": RayAttentionFusionModelTemporalResidual,
    "crossview_residual": RayAttentionFusionModelTemporalCrossviewResidual,
    "crossview_residual_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
    "crossview_residual_pp_visibility": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
}


def _build_model(
    j: int,
    n_views: int,
    d: int,
    n_temporal_layers: int,
    checkpoint: str | None,
    model_class: str = "temporal_residual",
    residual_hidden: int = 128,
):
    cls = MODEL_CLASSES[model_class]
    kwargs = {"j": j, "d": d, "n_views": n_views, "residual_hidden": residual_hidden}
    if model_class in {"crossview_residual", "crossview_residual_pp", "crossview_residual_pp_visibility"}:
        kwargs["n_st_layers"] = n_temporal_layers
    else:
        kwargs["n_temporal_layers"] = n_temporal_layers
    model = cls(**kwargs)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Allow loading checkpoints that contain extra heads (e.g. principal-point
        # correction) into the base cross-view/temporal model.
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate_variable_views(
    model,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    cameras,
    clip_len: int,
    device: torch.device,
    min_views: int = 2,
    max_views: int | None = None,
    k_values: list[int] | None = None,
    num_subsets_per_k: int | None = None,
    seed: int = 42,
    hardened: bool = False,
) -> dict:
    """Evaluate model for each view count in [min_views, max_views].

    For each view count k we sample up to ``num_subsets_per_k`` subsets of
    views.  When ``num_subsets_per_k`` is None we enumerate all C(V, k)
    subsets (only practical for small V).

    When ``k_values`` is provided it overrides ``min_views``/``max_views``
    and only the requested view counts are evaluated.  This is useful for
    producing a compact MPJPE@k table, e.g. for k = 2, 3, 4 plus full
    views.
    """
    if k_values is not None:
        k_values = sorted(set(k_values))
    else:
        if max_views is None:
            max_views = points_2d.shape[1]
        k_values = list(range(min_views, max_views + 1))

    raw = evaluate_mpjpe_at_k(
        model,
        points_2d,
        confidences,
        joints_3d,
        cameras,
        k_values=k_values,
        clip_len=clip_len,
        num_subsets_per_k=num_subsets_per_k,
        seed=seed,
        align="none",
        device=device,
        hardened=hardened,
    )
    # Preserve the legacy result shape expected by callers of this function.
    results = {}
    for k, res in raw.items():
        results[k] = {
            "mpjpe_at_k": res["mpjpe"],
            "mean_mm": res["mpjpe"],
            "std_mm": res["std_mm"],
            "n_subsets": res["n_subsets"],
            "temporal_jerk": res["temporal_jerk"],
        }
    return results


def _print_single(results, label="Variable view-count"):
    print(f"{label} benchmark results (MPJPE mm):")
    for k, res in results.items():
        print(f"  MPJPE@k={k:2d}: {res['mpjpe_at_k']:.4f} mm, "
              f"std={res['std_mm']:.4f} mm, subsets={res['n_subsets']}")


def _write_single_csv(results, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("k,mpjpe_at_k,mean_mm,std_mm,n_subsets,temporal_jerk\n")
        for k, res in results.items():
            f.write(f"{k},{res['mpjpe_at_k']:.4f},{res['mean_mm']:.4f},"
                    f"{res['std_mm']:.4f},{res['n_subsets']},{res.get('temporal_jerk', 0.0):.4f}\n")
    print(f"Saved CSV to {out_path}")


def _write_comparison_csv(results_v46, results_v47, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("k,v46_mpjpe_at_k,v47_mpjpe_at_k,delta_mm,delta_pct,"
                "v46_temporal_jerk,v47_temporal_jerk\n")
        for k in sorted(set(results_v46) | set(results_v47)):
            r46 = results_v46.get(k, {})
            r47 = results_v47.get(k, {})
            v46 = r46.get("mpjpe_at_k", float("nan"))
            v47 = r47.get("mpjpe_at_k", float("nan"))
            delta = v46 - v47
            delta_pct = (delta / v46 * 100.0) if v46 > 0 else 0.0
            j46 = r46.get("temporal_jerk", 0.0)
            j47 = r47.get("temporal_jerk", 0.0)
            f.write(f"{k},{v46:.4f},{v47:.4f},{delta:.4f},{delta_pct:.2f},"
                    f"{j46:.4f},{j47:.4f}\n")
    print(f"Saved comparison CSV to {out_path}")


def _print_per_dataset_results(results_by_dataset: Dict[str, dict]):
    """Print MPJPE@k for each dataset in ``results_by_dataset``."""
    for name, results in results_by_dataset.items():
        _print_single(results, label=f"Dataset: {name}")


def _print_cross_dataset_summary(results_by_dataset: Dict[str, dict]):
    """Print a per-k table across datasets and the domain gap."""
    if len(results_by_dataset) < 2:
        return
    all_k = sorted({k for results in results_by_dataset.values() for k in results})
    if not all_k:
        return
    print("\nCross-dataset MPJPE@k summary (mm):")
    header = f"{'dataset':>20} " + " ".join(f"k={k:>3}" for k in all_k)
    print(header)
    print("-" * len(header))
    for name, results in results_by_dataset.items():
        row = " ".join(
            f"{results.get(k, {}).get('mpjpe_at_k', float('nan')):>7.2f}" for k in all_k
        )
        print(f"{name:>20} {row}")

    print("\nDomain gap (max - min MPJPE@k across datasets, mm):")
    for k in all_k:
        vals = [results[k]["mpjpe_at_k"] for results in results_by_dataset.values() if k in results]
        if vals:
            print(f"  k={k:2d}: {max(vals) - min(vals):.4f} mm")


def _write_per_dataset_csv(results_by_dataset: Dict[str, dict], out_path: Path):
    """Write a CSV with one row per (dataset, view-count)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("dataset,k,mpjpe_at_k,mean_mm,std_mm,n_subsets,temporal_jerk\n")
        for name, results in results_by_dataset.items():
            for k, res in results.items():
                f.write(
                    f"{name},{k},{res['mpjpe_at_k']:.4f},{res['mean_mm']:.4f},"
                    f"{res['std_mm']:.4f},{res['n_subsets']},{res.get('temporal_jerk', 0.0):.4f}\n"
                )
    print(f"Saved per-dataset CSV to {out_path}")


def _write_per_dataset_comparison_csv(
    v46_by_dataset: Dict[str, dict],
    v47_by_dataset: Dict[str, dict],
    out_path: Path,
):
    """Write a CSV comparing v46 vs v47 per dataset and view-count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(
            "dataset,k,v46_mpjpe_at_k,v47_mpjpe_at_k,delta_mm,delta_pct,"
            "v46_temporal_jerk,v47_temporal_jerk\n"
        )
        for name in v46_by_dataset:
            r46 = v46_by_dataset[name]
            r47 = v47_by_dataset.get(name, {})
            for k in sorted(set(r46) | set(r47)):
                v46 = r46.get(k, {}).get("mpjpe_at_k", float("nan"))
                v47 = r47.get(k, {}).get("mpjpe_at_k", float("nan"))
                delta = v46 - v47
                delta_pct = (delta / v46 * 100.0) if v46 > 0 else 0.0
                j46 = r46.get(k, {}).get("temporal_jerk", 0.0)
                j47 = r47.get(k, {}).get("temporal_jerk", 0.0)
                f.write(
                    f"{name},{k},{v46:.4f},{v47:.4f},{delta:.4f},{delta_pct:.2f},"
                    f"{j46:.4f},{j47:.4f}\n"
                )
    print(f"Saved per-dataset comparison CSV to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to .npz with points_2d, confidences, joints_3d, camera_K/R/t")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Human-readable name for --dataset (defaults to file stem)")
    parser.add_argument("--dataset_manifest", type=str, default=None,
                        help="Path to a manifest file with one '<name> <path>' pair per line")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint for RayAttentionFusionModelTemporalResidual")
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument("--j", type=int, default=17)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--model_class", type=str, default="temporal_residual",
                        choices=list(MODEL_CLASSES.keys()) + ["omniview_v5"],
                        help="Model class to instantiate for the checkpoint")
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--k_values", type=int, nargs="+", default=None,
                        help=("Explicit list of view counts to evaluate, e.g. "
                              "'--k_values 2 3 4'. Overrides --min_views and "
                              "--max_views.  When omitted the full range is "
                              "evaluated."))
    parser.add_argument("--num_subsets_per_k", type=int, default=20,
                        help="Number of random view subsets to sample per view count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_frames", type=int, default=36,
                        help="Synthetic frame count when --dataset is not given")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional JSON path to save results")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional CSV path to save per-k or per-dataset summary")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to saved training config (JSON/YAML) for model_class=omniview_v5")
    parser.add_argument("--v46_checkpoint", type=str, default=None,
                        help="OmniMultiViewFusionV5 v46 checkpoint for side-by-side comparison")
    parser.add_argument("--v47_checkpoint", type=str, default=None,
                        help="OmniMultiViewFusionV5 v47 checkpoint for side-by-side comparison")
    parser.add_argument("--compare_v46_v47", action="store_true",
                        help="Compare v46 vs v47 by toggling use_v47_temporal_aggregation on the loaded model")
    args = parser.parse_args()

    if args.dataset is not None and args.dataset_manifest is not None:
        parser.error("--dataset and --dataset_manifest are mutually exclusive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load one or more datasets.
    # ------------------------------------------------------------------
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, list]] = {}
    if args.dataset_manifest is not None:
        for name, path in _load_dataset_manifest(args.dataset_manifest):
            datasets[name] = _load_npz_dataset(path)
    elif args.dataset is not None:
        name = args.dataset_name or Path(args.dataset).stem
        datasets[name] = _load_npz_dataset(args.dataset)
    else:
        points_2d, confidences, joints_3d, cameras = _make_synthetic_dataset(
            args.n_views, args.j, args.n_frames
        )
        datasets["synthetic"] = (points_2d, confidences, joints_3d, cameras)

    if not datasets:
        parser.error("No datasets to evaluate. Provide --dataset, --dataset_manifest, or omit both for a synthetic smoke test.")

    # ------------------------------------------------------------------
    # Shared helpers for running evaluation across loaded datasets.
    # ------------------------------------------------------------------
    def _run_eval_on(model, points_2d, confidences, joints_3d, cameras, n_views):
        return evaluate_variable_views(
            model,
            points_2d,
            confidences,
            joints_3d,
            cameras,
            clip_len=args.clip_len,
            device=device,
            min_views=args.min_views,
            max_views=args.max_views or n_views,
            k_values=args.k_values,
            num_subsets_per_k=args.num_subsets_per_k,
            seed=args.seed,
            hardened=isinstance(model.__class__.__name__, str)
            and "OmniMultiViewFusionV5" in model.__class__.__name__,
        )

    def _evaluate_all(model) -> Dict[str, dict]:
        results_by_dataset = {}
        for name, (points_2d, confidences, joints_3d, cameras) in datasets.items():
            n_views = points_2d.shape[1]
            results_by_dataset[name] = _run_eval_on(
                model, points_2d, confidences, joints_3d, cameras, n_views
            )
        return results_by_dataset

    def _handle_single_or_per_dataset_output(results_by_dataset):
        """Print and optionally save results for one or more datasets."""
        if len(results_by_dataset) == 1:
            name, results = next(iter(results_by_dataset.items()))
            _print_single(results, label=f"Dataset: {name}")
            if args.output_json:
                out_path = Path(args.output_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "mpjpe_at_k": {str(k): res["mpjpe_at_k"] for k, res in results.items()},
                            "per_k": {k: v for k, v in results.items()},
                        },
                        f,
                        indent=2,
                    )
                print(f"Saved results to {out_path}")
            if args.output_csv:
                _write_single_csv(results, Path(args.output_csv))
        else:
            _print_per_dataset_results(results_by_dataset)
            _print_cross_dataset_summary(results_by_dataset)
            if args.output_json:
                out_path = Path(args.output_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({"per_dataset": results_by_dataset}, f, indent=2)
                print(f"Saved results to {out_path}")
            if args.output_csv:
                _write_per_dataset_csv(results_by_dataset, Path(args.output_csv))

    # ------------------------------------------------------------------
    # v46 vs v47 comparison branch (OmniMultiViewFusionV5)
    # ------------------------------------------------------------------
    if args.model_class == "omniview_v5":
        n_joints = next(iter(datasets.values()))[2].shape[1]

        if args.v46_checkpoint and args.v47_checkpoint:
            cfg_v46_path = _resolve_config_path(args, args.v46_checkpoint)
            cfg_v47_path = _resolve_config_path(args, args.v47_checkpoint)
            if cfg_v46_path is None or cfg_v47_path is None:
                raise ValueError("--config is required for both v46 and v47 checkpoints")
            cfg_v46 = _load_config(cfg_v46_path)
            cfg_v47 = _load_config(cfg_v47_path)
            model_v46 = _build_omniview_v5_model(cfg_v46, args.v46_checkpoint, n_joints, 0, device)
            model_v47 = _build_omniview_v5_model(cfg_v47, args.v47_checkpoint, n_joints, 0, device)

            v46_by_dataset = _evaluate_all(model_v46)
            v47_by_dataset = _evaluate_all(model_v47)

            print("Variable view-count v46 vs v47 comparison (MPJPE mm):")
            for name in v46_by_dataset:
                print(f"\nDataset: {name}")
                r46 = v46_by_dataset[name]
                r47 = v47_by_dataset.get(name, {})
                print(f"{'k':>3} {'v46':>10} {'v47':>10} {'delta':>10} {'delta%':>8} {'jerk46':>10} {'jerk47':>10}")
                for k in sorted(set(r46) | set(r47)):
                    v46 = r46.get(k, {}).get("mpjpe_at_k", float("nan"))
                    v47 = r47.get(k, {}).get("mpjpe_at_k", float("nan"))
                    delta = v46 - v47
                    delta_pct = (delta / v46 * 100.0) if v46 > 0 else 0.0
                    j46 = r46.get(k, {}).get("temporal_jerk", 0.0)
                    j47 = r47.get(k, {}).get("temporal_jerk", 0.0)
                    print(f"{k:>3} {v46:>10.4f} {v47:>10.4f} {delta:>10.4f} {delta_pct:>7.2f}% "
                          f"{j46:>10.4f} {j47:>10.4f}")

            if args.output_json:
                out_path = Path(args.output_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({"v46": v46_by_dataset, "v47": v47_by_dataset}, f, indent=2)
                print(f"Saved results to {out_path}")

            if args.output_csv:
                _write_per_dataset_comparison_csv(
                    v46_by_dataset, v47_by_dataset, Path(args.output_csv)
                )

        else:
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for model_class=omniview_v5")
            cfg_path = _resolve_config_path(args, args.checkpoint)
            if cfg_path is None:
                raise ValueError("--config is required for model_class=omniview_v5")
            config = _load_config(cfg_path)
            model = _build_omniview_v5_model(config, args.checkpoint, n_joints, 0, device)

            if args.compare_v46_v47 and hasattr(model, "use_v47_temporal_aggregation"):
                # v46 mode: disable v47 temporal head.
                model.use_v47_temporal_aggregation = False
                v46_by_dataset = _evaluate_all(model)
                # v47 mode: enable temporal head.
                model.use_v47_temporal_aggregation = True
                v47_by_dataset = _evaluate_all(model)

                print("Variable view-count v46 vs v47 comparison (MPJPE mm):")
                for name in v46_by_dataset:
                    print(f"\nDataset: {name}")
                    r46 = v46_by_dataset[name]
                    r47 = v47_by_dataset.get(name, {})
                    print(f"{'k':>3} {'v46':>10} {'v47':>10} {'delta':>10} {'delta%':>8} {'jerk46':>10} {'jerk47':>10}")
                    for k in sorted(set(r46) | set(r47)):
                        v46 = r46.get(k, {}).get("mpjpe_at_k", float("nan"))
                        v47 = r47.get(k, {}).get("mpjpe_at_k", float("nan"))
                        delta = v46 - v47
                        delta_pct = (delta / v46 * 100.0) if v46 > 0 else 0.0
                        j46 = r46.get(k, {}).get("temporal_jerk", 0.0)
                        j47 = r47.get(k, {}).get("temporal_jerk", 0.0)
                        print(f"{k:>3} {v46:>10.4f} {v47:>10.4f} {delta:>10.4f} {delta_pct:>7.2f}% "
                              f"{j46:>10.4f} {j47:>10.4f}")

                if args.output_json:
                    out_path = Path(args.output_json)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w") as f:
                        json.dump({"v46": v46_by_dataset, "v47": v47_by_dataset}, f, indent=2)
                    print(f"Saved results to {out_path}")

                if args.output_csv:
                    _write_per_dataset_comparison_csv(
                        v46_by_dataset, v47_by_dataset, Path(args.output_csv)
                    )
            else:
                results_by_dataset = _evaluate_all(model)
                _handle_single_or_per_dataset_output(results_by_dataset)

    # ------------------------------------------------------------------
    # Legacy / ray-model branch
    # ------------------------------------------------------------------
    else:
        model = _build_model(args.j, args.n_views, args.d, args.n_temporal_layers, args.checkpoint, args.model_class, args.residual_hidden).to(device)

        results_by_dataset = _evaluate_all(model)
        _handle_single_or_per_dataset_output(results_by_dataset)


if __name__ == "__main__":
    main()
