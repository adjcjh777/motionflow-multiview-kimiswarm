"""Cross-dataset evaluation of a v25 OmniMultiViewFusionV5 checkpoint.

Loads a checkpoint saved by ``experiments/train_omniview_fusion_v5_webbridge_multi.py``
(normally trained on H36M true-GT v2) and evaluates it on a different WebBridge
dataset (AIST++, Shelf/Campus, MPI-INF-3DHP detected-2D, etc.).

The script uses :class:`WebBridgeCanonical17Dataset` so that the target data are
mapped to the canonical 17-joint H36M skeleton and padded to the 14-view layout
that the v25 checkpoint expects.  A ``view_mask`` is passed to the model so that
padded views are ignored.

Usage
-----
    # AIST++ test split (metrics)
    python experiments/eval_v25_cross_dataset.py \
        --checkpoint outputs/ablations/v25_true_gt_v2_medium_a800.pth \
        --dataset_name aist \
        --split configs/splits/aistpp_train_val_test.yaml \
        --out_json outputs/eval_v25_true_gt_v2_aistpp.json

    # Shelf/Campus detected .npz files (metrics)
    python experiments/eval_v25_cross_dataset.py \
        --checkpoint outputs/ablations/v25_true_gt_v2_medium_a800.pth \
        --dataset_name shelf \
        --input data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz \
                data/webbridge/shelf_campus_detected/campus_seq1_val_detected_m.npz \
        --out_json outputs/eval_v25_true_gt_v2_shelf_campus.json

    # MPI-INF-3DHP detected-2D prediction-only (no GT metrics)
    python experiments/eval_v25_cross_dataset.py \
        --checkpoint outputs/ablations/v25_true_gt_v2_medium_a800.pth \
        --dataset_name mpi \
        --split configs/splits/mpi_inf_3dhp_detected_2d_baseline.yaml \
        --infer_only \
        --out_json outputs/eval_v25_true_gt_v2_mpi.json \
        --out_npz outputs/eval_v25_true_gt_v2_mpi_predictions.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from motionflow_mv.data.webbridge_mixed_dataset import (  # noqa: E402
    DATASET_IDS,
    WebBridgeCanonical17Dataset,
)
from motionflow_mv.eval.metrics import compute_all_metrics  # noqa: E402
from motionflow_mv.training.trainer_v2 import EMA  # noqa: E402
from experiments.train_omniview_fusion_v5_webbridge_multi import (  # noqa: E402
    build_model_from_args,
)


def _collate_with_mask(batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    # The v25 true-GT v2 checkpoint was trained on H36M only (num_domains=1),
    # so the domain embedding only has an entry for H36M.  Force domain_id=0
    # for cross-dataset evaluation so the embedding lookup stays valid.
    dataset_ids = torch.zeros(len(batch), dtype=torch.long)
    view_mask = torch.stack([b[6] for b in batch], dim=0)
    return x, y, K, R, t, dataset_ids, view_mask


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state
    if isinstance(state, dict) and "model" in state:
        model_state = state["model"]
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")

    ema_state = None
    if isinstance(state, dict):
        ema_state = state.get("ema")
    if ema_state is not None:
        ema = EMA(model, decay=ema_state.get("decay", 0.999), update_every=ema_state.get("update_every", 1))
        ema.load_state_dict(ema_state)
        ema.apply_shadow(model)
        print("Applied EMA shadow weights for evaluation.")
    else:
        print("No EMA state found; using online checkpoint weights.")

    return model


def _build_model_args(config: Dict[str, Any]) -> Namespace:
    """Convert a saved training config into the Namespace expected by the trainer builder."""
    model_args = Namespace(**config)
    # Defensive defaults for keys that older configs might omit.
    defaults = {
        "d": 128,
        "residual_hidden": 256,
        "n_st_layers": 3,
        "graph_num_layers": 1,
        "n_joint_layers": 1,
        "n_heads": 4,
        "epipolar_loss_weight": 0.05,
        "num_domains": 8,
        "use_full_precision_dlt": False,
        "use_robust_dlt_reweight": False,
        "use_irls_reweight": False,
        "use_domain_embedding": False,
    }
    for key, value in defaults.items():
        if not hasattr(model_args, key):
            setattr(model_args, key, value)
    return model_args


def evaluate_file(
    model: torch.nn.Module,
    device: torch.device,
    npz_path: str,
    dataset_name: str,
    clip_len: int,
    batch_size: int,
    val_stride: int,
    infer_only: bool = False,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    dataset = WebBridgeCanonical17Dataset(
        npz_path,
        dataset_name=dataset_name,
        clip_len=clip_len,
        n_samples=None,
        stride=val_stride,
        return_view_mask=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_with_mask,
        num_workers=0,
    )

    preds_list: List[np.ndarray] = []
    gts_list: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, yb, K, R, t, domain_id, view_mask in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            domain_id = domain_id.to(device)
            view_mask = view_mask.to(device)

            out = model(
                xb,
                K=K,
                R=R,
                t=t,
                view_mask=view_mask,
                domain_id=domain_id,
            )
            pred = out[0]
            preds_list.append(pred.cpu().numpy())
            gts_list.append(yb.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0).reshape(-1, preds_list[0].shape[-2], 3)
    gts = np.concatenate(gts_list, axis=0).reshape(-1, gts_list[0].shape[-2], 3)

    # Determine whether ground truth is usable.  The canonical loader always
    # returns a y tensor, but for prediction-only datasets it may be all zeros.
    has_gt = not infer_only and not np.allclose(gts, 0.0)

    result: Dict[str, Any] = {
        "npz": npz_path,
        "dataset_name": dataset_name,
        "n_frames": int(preds.shape[0]),
    }
    if has_gt:
        report = compute_all_metrics(preds * 1000.0, gts * 1000.0)
        result["mpjpe_mm"] = float(report["mpjpe"])
        result["pa_mpjpe_mm"] = float(report["pa_mpjpe"])
    else:
        result["mpjpe_mm"] = None
        result["pa_mpjpe_mm"] = None

    return result, preds, gts


def _load_yaml_split(split_path: str) -> List[Tuple[str, str]]:
    """Return a list of (npz_path, dataset_name) from a YAML split file.

    The YAML may contain ``train/val/test`` keys with optional matching
    ``train_names/val_names/test_names`` lists.  If names are absent, the
    caller must supply a default dataset name via ``--dataset_name``.
    """
    with open(split_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    items: List[Tuple[str, str]] = []
    for split in ("train", "val", "test"):
        paths = cfg.get(split, []) or []
        names = cfg.get(f"{split}_names", []) or []
        for i, p in enumerate(paths):
            name = names[i] if i < len(names) else None
            items.append((p, name))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate v25 checkpoint on a cross-dataset WebBridge split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ablations/v25_true_gt_v2_medium_a800.pth",
        help="Path to v25 .pth checkpoint",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Path to training config JSON sidecar (default: <checkpoint>.config.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="YAML split manifest with train/val/test lists",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="*",
        default=None,
        help="One or more canonical .npz files to evaluate",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        choices=list(DATASET_IDS.keys()),
        help="WebBridge dataset name (h36m, mpi, aist, shelf, campus). Required unless split provides names.",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--val_stride",
        type=int,
        default=13,
        help="Stride for test clips. Use 1 for dense eval, 13 to match prior v25 test reporting.",
    )
    parser.add_argument(
        "--infer_only",
        action="store_true",
        help="Skip GT metrics and only save predictions",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="outputs/eval_v25_true_gt_v2_cross_dataset.json",
        help="Where to write the JSON result",
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        default=None,
        help="Optional .npz path to save per-file predictions (keys = file stems)",
    )
    args = parser.parse_args()

    if args.split is None and not args.input:
        parser.error("Provide either --split or --input")

    # Gather files to evaluate.
    files: List[Tuple[str, str]] = []
    if args.split:
        files.extend(_load_yaml_split(args.split))
    if args.input:
        for p in args.input:
            files.append((p, args.dataset_name))

    if not args.dataset_name:
        # If no global dataset_name, the split must supply names for every file.
        missing_names = [p for p, name in files if name is None]
        if missing_names:
            parser.error(
                f"--dataset_name is required when split does not provide names. "
                f"Files without names: {missing_names[:5]}"
            )

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config_path = args.config_json or Path(args.checkpoint).with_suffix(".config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    n_views = config.get("n_views", 14)
    j = config.get("j", 17)
    print(f"Checkpoint config: n_views={n_views}, j={j}")

    model_args = _build_model_args(config)
    model = build_model_from_args(model_args, n_joints=j, n_views=n_views, device=device)
    _load_checkpoint(model, args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    per_file_results: List[Dict[str, Any]] = []
    per_file_preds: Dict[str, np.ndarray] = {}

    for npz_path, dataset_name in files:
        default_name = args.dataset_name or "h36m"
        dataset_name = dataset_name or default_name
        if dataset_name not in DATASET_IDS:
            print(f"[SKIP] Unknown dataset_name '{dataset_name}' for {npz_path}")
            continue

        if not Path(npz_path).exists():
            print(f"[SKIP] Not found: {npz_path}")
            continue

        print(f"\nEvaluating {Path(npz_path).name} (dataset={dataset_name})...")
        result, preds, gts = evaluate_file(
            model,
            device,
            npz_path,
            dataset_name=dataset_name,
            clip_len=args.clip_len,
            batch_size=args.batch_size,
            val_stride=args.val_stride,
            infer_only=args.infer_only,
        )
        per_file_results.append(result)
        per_file_preds[Path(npz_path).stem] = preds
        status = f"  MPJPE={result['mpjpe_mm']:.2f} mm" if result["mpjpe_mm"] is not None else "  (predictions only)"
        print(f"  frames={result['n_frames']}{status}")

    # Aggregate over files that have GT.
    gt_results = [r for r in per_file_results if r["mpjpe_mm"] is not None]
    aggregate: Dict[str, Any] = {"n_files": len(per_file_results), "n_files_with_gt": len(gt_results)}
    if gt_results:
        mpjpe_values = [r["mpjpe_mm"] for r in gt_results]
        pa_values = [r["pa_mpjpe_mm"] for r in gt_results]
        aggregate["mean_mpjpe_mm"] = float(np.mean(mpjpe_values))
        aggregate["mean_pa_mpjpe_mm"] = float(np.mean(pa_values))
        aggregate["std_mpjpe_mm"] = float(np.std(mpjpe_values))
        aggregate["std_pa_mpjpe_mm"] = float(np.std(pa_values))
        print("\n--- Summary ---")
        print(f"Mean MPJPE:     {aggregate['mean_mpjpe_mm']:.2f} mm")
        print(f"Mean PA-MPJPE:  {aggregate['mean_pa_mpjpe_mm']:.2f} mm")

    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "config_json": str(config_path),
        "infer_only": args.infer_only,
        "aggregate": aggregate,
        "per_file": per_file_results,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON results to {out_path}")

    if args.out_npz:
        npz_path = Path(args.out_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **per_file_preds)
        print(f"Saved predictions to {npz_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "dataset", "n_frames", "mpjpe_mm", "pa_mpjpe_mm"])
        writer.writeheader()
        for r in per_file_results:
            writer.writerow({
                "file": r["npz"],
                "dataset": r["dataset_name"],
                "n_frames": r["n_frames"],
                "mpjpe_mm": r["mpjpe_mm"] if r["mpjpe_mm"] is not None else "",
                "pa_mpjpe_mm": r["pa_mpjpe_mm"] if r["pa_mpjpe_mm"] is not None else "",
            })
    print(f"Saved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()
