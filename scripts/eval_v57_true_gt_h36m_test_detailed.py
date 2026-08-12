"""Detailed evaluation of a v57 OmniMultiViewFusionV5 checkpoint on true-GT H36M test subjects.

Runs dense (stride=1) evaluation by default and writes all scalar metrics plus
per-joint MPJPE/PA-MPJPE arrays for S9, S11, and their combined average.

Usage
-----
    python scripts/eval_v57_true_gt_h36m_test_detailed.py \
        --checkpoint outputs/ablations/v57_true_gt_medium_a800.pth \
        --config_json outputs/ablations/v57_true_gt_medium_a800.config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from motionflow_mv.data.webbridge_mixed_dataset import WebBridgeCanonical17Dataset
from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.training.trainer_v2 import EMA
from experiments.train_omniview_fusion_v5_webbridge_multi import (  # noqa: E402
    build_model_from_args,
)


def _collate_with_mask(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    dataset_ids = torch.tensor([b[5] for b in batch], dtype=torch.long)
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

    # The trainer evaluates with EMA weights; apply the saved EMA shadow if present.
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


def evaluate_subject(
    model: torch.nn.Module,
    device: torch.device,
    npz_path: str,
    clip_len: int,
    batch_size: int,
    val_stride: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    dataset = WebBridgeCanonical17Dataset(
        npz_path,
        dataset_name="h36m",
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

    preds, gts = [], []
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
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    report = compute_all_metrics(preds, gts)
    return report, preds, gts


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
        "num_domains": 2,
        "use_full_precision_dlt": False,
        "use_robust_dlt_reweight": False,
        "use_irls_reweight": False,
        "use_domain_embedding": False,
    }
    for key, value in defaults.items():
        if not hasattr(model_args, key):
            setattr(model_args, key, value)
    return model_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate v57 checkpoint on true-GT H36M test subjects S9/S11",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ablations/v57_true_gt_medium_a800.pth",
        help="Path to v57 .pth checkpoint",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Path to training config JSON sidecar (default: <checkpoint>.config.json)",
    )
    parser.add_argument(
        "--s9",
        type=str,
        default="data/h36m_true_gt/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
        help="Path to S9 test .npz",
    )
    parser.add_argument(
        "--s11",
        type=str,
        default="data/h36m_true_gt/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
        help="Path to S11 test .npz",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--val_stride",
        type=int,
        default=1,
        help="Stride for test clips. Default 1 for detailed (dense) eval; use 13 to match prior v57 test reporting.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="outputs/eval_v57_true_gt_h36m_test_detailed.json",
        help="Where to write the JSON result",
    )
    args = parser.parse_args()

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

    def _convert_for_json(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, dict):
            return {k: _convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert_for_json(v) for v in obj]
        return obj

    results: Dict[str, Any] = {}
    for name, path in [("S9", args.s9), ("S11", args.s11)]:
        print(f"\nEvaluating {name}: {path}")
        report, preds, gts = evaluate_subject(
            model,
            device,
            path,
            clip_len=args.clip_len,
            batch_size=args.batch_size,
            val_stride=args.val_stride,
        )
        print(f"{name} MPJPE: {report['mpjpe']:.2f} mm")
        print(f"{name} PA-MPJPE: {report['pa_mpjpe']:.2f} mm")
        # Store all scalar metrics plus per-joint arrays for detailed analysis.
        subject_results: Dict[str, Any] = {
            "n_frames": preds.shape[0],
            "preds_shape": list(preds.shape),
            "gts_shape": list(gts.shape),
        }
        for k, v in report.items():
            subject_results[k] = _convert_for_json(v)
        results[name] = subject_results

    # Combined summary (equal weight to S9 and S11).
    combined: Dict[str, Any] = {"n_frames": (results["S9"]["n_frames"] + results["S11"]["n_frames"])}
    scalar_keys = [k for k in results["S9"].keys() if isinstance(results["S9"][k], (int, float)) and k not in {"n_frames"}]
    for k in scalar_keys:
        combined[k] = (results["S9"][k] + results["S11"][k]) / 2.0
    # Average per-joint arrays jointly across subjects.
    for k in ("per_joint_mpjpe", "per_joint_pa_mpjpe"):
        arr_s9 = np.asarray(results["S9"][k])
        arr_s11 = np.asarray(results["S11"][k])
        n_s9 = results["S9"]["n_frames"]
        n_s11 = results["S11"]["n_frames"]
        combined[k] = ((arr_s9 * n_s9 + arr_s11 * n_s11) / (n_s9 + n_s11)).tolist()
    results["combined"] = combined

    print("\n--- Summary ---")
    for name in ("S9", "S11", "combined"):
        print(
            f"{name}: MPJPE={results[name]['mpjpe']:.2f}mm  "
            f"PA-MPJPE={results[name]['pa_mpjpe']:.2f}mm  "
            f"Root-Rel={results[name].get('root_rel_mpjpe', float('nan')):.2f}mm  "
            f"PCK@50mm={results[name].get('pck@50mm', float('nan')):.3f}  "
            f"PCK@150mm={results[name].get('pck@150mm', float('nan')):.3f}"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
