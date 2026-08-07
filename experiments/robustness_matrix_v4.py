"""Robustness matrix harness for OmniMultiViewFusion v4.

Generalises the existing v2/v3 robustness matrix to support v4 (when it is
available), variable-view inference, and per-joint metrics.  It can evaluate
any fixed-view OmniMultiViewFusion model (v2/v3/v4) and writes a CSV compatible
with ``docs/tables/icra2027/robustness_matrix.md``.

Usage
-----
    # CPU smoke test (uses OmniMultiViewFusionV3 fallback if v4 is not present)
    python experiments/robustness_matrix_v4.py --smoke

    # Evaluate a trained v4 checkpoint
    python experiments/robustness_matrix_v4.py \
        --model v4 \
        --checkpoint outputs/omniview_fusion_v4_best.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --output_dir outputs/robustness_matrix_v4
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.variable_view_inference import VariableViewInferenceWrapper


# ---------------------------------------------------------------------------
# Re-use as much as possible from the v3 eval driver.
# ---------------------------------------------------------------------------
_V3_EVAL_PATH = Path(__file__).with_name("eval_omniview_fusion_v3_mpiinf3dhp.py")
spec = importlib.util.spec_from_file_location("eval_omniview_fusion_v3_mpiinf3dhp", _V3_EVAL_PATH)
eval_v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_v3)

TemporalClipDataset = eval_v3.TemporalClipDataset
SyntheticSmokeDataset = eval_v3.SyntheticSmokeDataset
_make_synthetic_cameras = eval_v3._make_synthetic_cameras
collate_fn = eval_v3.collate_fn
build_v3_model = eval_v3.build_model
load_checkpoint = eval_v3.load_checkpoint
evaluate_clean = eval_v3.evaluate_clean
evaluate_perturbed = eval_v3.evaluate_perturbed
robustness_conditions = eval_v3.robustness_conditions
so3_exp = eval_v3.so3_exp
corrupt_intrinsics = eval_v3.corrupt_intrinsics
corrupt_extrinsics = eval_v3.corrupt_extrinsics


# ---------------------------------------------------------------------------
# Model registry with v4 support.
# ---------------------------------------------------------------------------
def _try_import_v4():
    try:
        from motionflow_mv.fusion.omniview_fusion_v4 import (  # type: ignore
            OmniMultiViewFusionV4,
        )

        return OmniMultiViewFusionV4
    except Exception:  # pragma: no cover
        return None


OmniMultiViewFusionV4 = _try_import_v4()


def _build_v2_model(args: argparse.Namespace, n_views: int, j: int) -> torch.nn.Module:
    from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2

    return OmniMultiViewFusionV2(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        graph_num_layers=args.graph_num_layers,
        n_joint_layers=args.n_joint_layers,
        return_pp_delta=False,
        return_covariance=True,
    )


def _build_v3_model(args: argparse.Namespace, n_views: int, j: int) -> torch.nn.Module:
    return build_v3_model(args, n_views, j)


def _build_v4_model(args: argparse.Namespace, n_views: int, j: int) -> torch.nn.Module:
    if OmniMultiViewFusionV4 is None:
        raise RuntimeError(
            "OmniMultiViewFusionV4 is not available. "
            "Install/merge T01 or run with --model v3."
        )
    return OmniMultiViewFusionV4(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        graph_num_layers=args.graph_num_layers,
        n_joint_layers=args.n_joint_layers,
        return_pp_delta=False,
        return_covariance=True,
        use_multiscale_fusion=args.use_multiscale_fusion,
        use_camera_conditioning=args.use_camera_conditioning,
        use_epipolar_bias=args.use_epipolar_bias,
        use_context_visibility=args.use_context_visibility,
        use_skeleton_residual=args.use_skeleton_residual,
        use_kinematic_refiner=args.use_kinematic_refiner,
        use_adaptive_view_selection=args.use_adaptive_view_selection,
        use_rotation_correction=args.use_rotation_correction,
        use_entropy_regularization=args.use_entropy_regularization,
    )


MODEL_BUILDERS = {
    "v2": _build_v2_model,
    "v3": _build_v3_model,
    "v4": _build_v4_model,
}


def build_model(args: argparse.Namespace, n_views: int, j: int) -> torch.nn.Module:
    requested = args.model
    if requested == "v4" and OmniMultiViewFusionV4 is None:
        print("WARNING: OmniMultiViewFusionV4 is not available; falling back to v3 for smoke/eval.")
        requested = "v3"
    return MODEL_BUILDERS[requested](args, n_views, j)


# ---------------------------------------------------------------------------
# Extended robustness conditions.
# Includes the v3 calibration perturbations plus explicit view/joint dropouts.
# ---------------------------------------------------------------------------
def _base_conditions(smoke: bool = False) -> Dict[str, Dict[str, Any]]:
    conditions = {
        "clean": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_0.5_deg": {"rot_std": 0.5, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_1.0_deg": {"rot_std": 1.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_5mm": {"rot_std": 0.0, "trans_std": 0.005, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_10mm": {"rot_std": 0.0, "trans_std": 0.010, "focal_err": 0.0, "cxcy_err": 0.0},
        "focal_1pct": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.01, "cxcy_err": 0.0},
        "focal_2pct": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.02, "cxcy_err": 0.0},
        "cxcy_3px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 3.0},
        "cxcy_5px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 5.0},
        "view_dropout_0.2": {"view_dropout": 0.2},
        "view_dropout_0.4": {"view_dropout": 0.4},
        "joint_dropout_0.2": {"joint_dropout": 0.2},
        "joint_dropout_0.4": {"joint_dropout": 0.4},
    }
    if smoke:
        conditions = {
            "clean": conditions["clean"],
            "rot_0.5_deg": conditions["rot_0.5_deg"],
            "focal_1pct": conditions["focal_1pct"],
            "cxcy_3px": conditions["cxcy_3px"],
        }
    return conditions


def _apply_view_dropout(x: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    """Zero out confidence/pixels of a random ``rate`` fraction of views."""
    if rate <= 0.0:
        return x
    B, T, V, J, _ = x.shape
    n_drop = max(1, int(round(V * rate)))
    x = x.clone()
    for b in range(B):
        perm = torch.randperm(V, generator=generator, device=generator.device)
        drop = perm[:n_drop]
        x[b, :, drop, :, :] = 0.0
    return x


def _apply_joint_dropout(x: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    """Zero out confidence/pixels of a random ``rate`` fraction of joints across all views."""
    if rate <= 0.0:
        return x
    B, T, V, J, _ = x.shape
    n_drop = max(1, int(round(J * rate)))
    x = x.clone()
    for b in range(B):
        perm = torch.randperm(J, generator=generator, device=generator.device)
        drop = perm[:n_drop]
        x[b, :, :, drop, :] = 0.0
    return x


def evaluate_condition(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    condition: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Evaluate one robustness condition and return a full metrics report."""
    model.eval()
    generator = torch.Generator(device=str(device) if device.type != "meta" else "cpu")
    generator.manual_seed(seed)

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)

            if "view_dropout" in condition:
                xb = _apply_view_dropout(xb, condition["view_dropout"], generator)
            if "joint_dropout" in condition:
                xb = _apply_joint_dropout(xb, condition["joint_dropout"], generator)

            K_in, R_in, t_in = K, R, t
            if "rot_std" in condition:
                # Calibration-only conditions are handled inside evaluate_perturbed,
                # so skip here if no rotation/translation/focal/cxcy keys.
                pass

            out = model(xb, K=K_in, R=R_in, t=t_in)
            pred = out[0]
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def run_calibration_condition(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    cfg: Dict[str, float],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run a v3-style calibration perturbation and return a full metrics report."""
    report = evaluate_perturbed(model, dataset, cfg, batch_size, device)
    # evaluate_perturbed already returns scalar metrics in mm.
    return report


# ---------------------------------------------------------------------------
# Variable-view evaluation.
# ---------------------------------------------------------------------------
def evaluate_variable_views(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    n_views: int,
    device: torch.device,
    min_views: int,
    max_views: int | None,
    num_subsets_per_k: int,
    seed: int,
) -> Dict[str, Any]:
    """Variable-view MPJPE@k with full per-joint metrics."""
    wrapper = VariableViewInferenceWrapper(model)
    rng = np.random.default_rng(seed)
    max_views = min(max_views or n_views, n_views)

    results: Dict[str, Any] = {}
    for k in range(min_views, max_views + 1):
        subsets = list(combinations(range(n_views), k))
        if len(subsets) > num_subsets_per_k:
            idx = rng.choice(len(subsets), size=num_subsets_per_k, replace=False)
            subsets = [subsets[i] for i in idx]

        per_subset_reports: List[Dict[str, Any]] = []
        for subset in subsets:
            preds, gts = [], []
            for xb, yb, K, R, t in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                K = K.to(device)
                R = R.to(device)
                t = t.to(device)
                with torch.no_grad():
                    pred = wrapper(xb, K, R, t, active_views=list(subset))[0]
                preds.append(pred.cpu().numpy())
                gts.append(yb.cpu().numpy())

            preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
            gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
            report = compute_all_metrics(preds, gts)
            per_subset_reports.append(report)

        mean_mpjpe = float(np.mean([r["mpjpe"] for r in per_subset_reports]))
        std_mpjpe = float(np.std([r["mpjpe"] for r in per_subset_reports]))
        mean_pa = float(np.mean([r["pa_mpjpe"] for r in per_subset_reports]))
        std_pa = float(np.std([r["pa_mpjpe"] for r in per_subset_reports]))
        # Average per-joint arrays across subsets.
        per_joint_mpjpe = np.mean(np.stack([r["per_joint_mpjpe"] for r in per_subset_reports], axis=0), axis=0)
        per_joint_pa = np.mean(np.stack([r["per_joint_pa_mpjpe"] for r in per_subset_reports], axis=0), axis=0)

        results[str(k)] = {
            "mean_mpjpe_mm": mean_mpjpe,
            "std_mpjpe_mm": std_mpjpe,
            "mean_pa_mpjpe_mm": mean_pa,
            "std_pa_mpjpe_mm": std_pa,
            "n_subsets": len(per_subset_reports),
            "per_joint_mpjpe_mm": per_joint_mpjpe,
            "per_joint_pa_mpjpe_mm": per_joint_pa,
        }
    return results


# ---------------------------------------------------------------------------
# Result I/O.
# ---------------------------------------------------------------------------
def _scalar_only(report: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}


def _per_joint_keys(report: Dict[str, Any]) -> List[str]:
    """Return keys in a metrics report that contain per-joint arrays."""
    return [k for k in report.keys() if k.startswith("per_joint_") or k.endswith("_per_joint")]


def _build_csv_rows(
    robustness: Dict[str, Dict[str, Any]],
    variable_views: Dict[str, Any],
    n_joints: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Robustness matrix rows (markdown compatible).
    for name, r in robustness.items():
        rows.append({
            "section": "robustness",
            "condition": name,
            "mpjpe": f"{r['mpjpe']:.2f}",
            "pa_mpjpe": f"{r['pa_mpjpe']:.2f}",
            "pck@50mm": f"{r['pck@50mm']:.3f}",
            "pck@100mm": f"{r['pck@100mm']:.3f}",
            "pck@150mm": f"{r['pck@150mm']:.3f}",
            "pck_auc": f"{r['pck_auc']:.3f}",
        })

    # Variable-view rows.
    for k, r in variable_views.items():
        rows.append({
            "section": "variable_views",
            "condition": f"k={k}",
            "mpjpe": f"{r['mean_mpjpe_mm']:.2f}",
            "pa_mpjpe": f"{r['mean_pa_mpjpe_mm']:.2f}",
            "pck@50mm": "",
            "pck@100mm": "",
            "pck@150mm": "",
            "pck_auc": "",
            "mpjpe_std": f"{r['std_mpjpe_mm']:.2f}",
            "pa_mpjpe_std": f"{r['std_pa_mpjpe_mm']:.2f}",
            "n_subsets": r["n_subsets"],
        })

    # Per-joint metric rows (one row per condition per metric).
    for cond_name, r in robustness.items():
        for metric_key in _per_joint_keys(r):
            short = metric_key.replace("per_joint_", "").replace("_per_joint", "")
            row: Dict[str, Any] = {
                "section": "per_joint",
                "condition": f"{cond_name}_{short}",
                "mpjpe": "",
                "pa_mpjpe": "",
                "pck@50mm": "",
                "pck@100mm": "",
                "pck@150mm": "",
                "pck_auc": "",
            }
            arr = r[metric_key]
            for j_idx in range(n_joints):
                row[f"joint_{j_idx}"] = f"{arr[j_idx]:.2f}"
            rows.append(row)

    return rows


def write_csv(csv_path: Path, rows: List[Dict[str, Any]], n_joints: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "condition",
        "mpjpe",
        "pa_mpjpe",
        "pck@50mm",
        "pck@100mm",
        "pck@150mm",
        "pck_auc",
    ]
    fieldnames += ["mpjpe_std", "pa_mpjpe_std", "n_subsets"]
    fieldnames += [f"joint_{j}" for j in range(n_joints)]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(json_path: Path, payload: Dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_convert(payload), f, indent=2)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness matrix harness for OmniMultiViewFusion v4.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["v2", "v3", "v4"],
        default="v4",
        help="Model family to evaluate (v4 falls back to v3 if not present).",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="Path to MPI-INF-3DHP .npz")
    parser.add_argument("--smoke", action="store_true", help="CPU/GPU smoke test on synthetic data")
    parser.add_argument("--output_dir", type=str, default="outputs/robustness_matrix_v4", help="Output directory")
    # Model architecture.
    parser.add_argument("--d", type=int, default=64, help="Feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--n_joint_layers", type=int, default=0, help="Dense joint-level transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    # v3 flags.
    parser.add_argument("--use_multiscale_fusion", action="store_true")
    parser.add_argument("--use_camera_conditioning", action="store_true")
    parser.add_argument("--use_epipolar_bias", action="store_true")
    # v4 toggles.
    parser.add_argument("--use_context_visibility", action="store_true")
    parser.add_argument("--use_skeleton_residual", action="store_true")
    parser.add_argument("--use_kinematic_refiner", action="store_true")
    parser.add_argument("--use_adaptive_view_selection", action="store_true")
    parser.add_argument("--use_rotation_correction", action="store_true")
    parser.add_argument("--use_entropy_regularization", action="store_true")
    # Evaluation.
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--run_variable_views", action="store_true", help="Run variable-view MPJPE@k")
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--num_subsets_per_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Device; defaults to cuda if available")
    args = parser.parse_args()

    if args.smoke:
        args.clip_len = 9
        args.batch_size = 2
        args.num_subsets_per_k = 2
        args.run_variable_views = True
        if args.checkpoint is None:
            args.checkpoint = "__smoke__"
        # Default to a representative v3/v4 stack in smoke mode.
        args.use_multiscale_fusion = True
        args.use_camera_conditioning = True
        args.use_epipolar_bias = True
        args.use_context_visibility = True
        args.use_skeleton_residual = True
    else:
        if args.checkpoint is None or args.dataset is None:
            parser.error("--checkpoint and --dataset are required unless --smoke is set")

    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.smoke:
        print("Smoke mode: using synthetic dataset")
        n_views = 4
        n_joints = 17
        K, R, t = _make_synthetic_cameras(n_views=n_views)
        dataset = SyntheticSmokeDataset(K, R, t, n_frames=60, n_joints=n_joints, clip_len=args.clip_len)
    else:
        data = np.load(args.dataset)
        n_views = int(data["camera_K"].shape[0])
        n_joints = int(data["points_2d"].shape[2])
        dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args, n_views=n_views, j=n_joints).to(device)
    if args.checkpoint and args.checkpoint != "__smoke__":
        load_checkpoint(model, args.checkpoint)
    else:
        print("No checkpoint provided; using freshly initialised model for smoke test")
    model.eval()

    # ------------------------------------------------------------------
    # Clean evaluation
    # ------------------------------------------------------------------
    print("Clean evaluation...")
    clean_report = evaluate_clean(model, loader, device)
    clean_summary = _scalar_only(clean_report)
    print(
        f"Clean: MPJPE={clean_summary['mpjpe']:.2f}mm "
        f"PA-MPJPE={clean_summary['pa_mpjpe']:.2f}mm"
    )

    results: Dict[str, Any] = {"clean": clean_summary}

    # ------------------------------------------------------------------
    # Robustness matrix
    # ------------------------------------------------------------------
    conditions = _base_conditions(smoke=args.smoke)
    robustness: Dict[str, Dict[str, Any]] = {}
    print("Calibration-robustness matrix...")
    for name, cfg in conditions.items():
        if name == "clean":
            robustness[name] = clean_report
            continue
        if any(k in cfg for k in ("rot_std", "trans_std", "focal_err", "cxcy_err")):
            report = run_calibration_condition(model, dataset, cfg, args.batch_size, device)
        else:
            report = evaluate_condition(model, loader, device, cfg, args.seed)
        robustness[name] = report
        summary = _scalar_only(report)
        print(
            f"{name}: MPJPE={summary['mpjpe']:.2f}mm "
            f"PA-MPJPE={summary['pa_mpjpe']:.2f}mm"
        )
    results["robustness"] = {k: _scalar_only(v) for k, v in robustness.items()}

    # Per-joint metrics for every condition (for JSON/CSV).
    per_joint_all = {}
    for name, report in robustness.items():
        per_joint_all[name] = {
            "mpjpe": report["per_joint_mpjpe"],
            "pa_mpjpe": report["per_joint_pa_mpjpe"],
            "pck@50mm": report["pck@50mm_per_joint"],
            "pck@100mm": report["pck@100mm_per_joint"],
            "pck@150mm": report["pck@150mm_per_joint"],
            "pck_auc": report["pck_auc_per_joint"],
        }
    results["per_joint"] = per_joint_all

    # ------------------------------------------------------------------
    # Variable-view curve
    # ------------------------------------------------------------------
    variable_views: Dict[str, Any] = {}
    if args.run_variable_views:
        print("Variable-view MPJPE@k curve...")
        variable_views = evaluate_variable_views(
            model,
            loader,
            n_views=n_views,
            device=device,
            min_views=args.min_views,
            max_views=args.max_views,
            num_subsets_per_k=args.num_subsets_per_k,
            seed=args.seed,
        )
        results["variable_views"] = variable_views
        for k, r in variable_views.items():
            print(
                f"  k={k}: MPJPE={r['mean_mpjpe_mm']:.2f}+/-{r['std_mpjpe_mm']:.2f}mm "
                f"PA-MPJPE={r['mean_pa_mpjpe_mm']:.2f}+/-{r['std_pa_mpjpe_mm']:.2f}mm "
                f"n={r['n_subsets']}"
            )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "robustness_matrix_v4.json"
    write_json(json_path, results)
    print(f"Saved JSON -> {json_path}")

    rows = _build_csv_rows(robustness, variable_views, n_joints)
    csv_path = output_dir / "robustness_matrix_v4.csv"
    write_csv(csv_path, rows, n_joints)
    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
