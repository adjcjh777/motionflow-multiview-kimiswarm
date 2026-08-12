"""Evaluate a v25 OmniMultiViewFusionV5 checkpoint on true-GT H36M test subjects.

The v25 checkpoint trained through the WebBridge mixed loader was saved with
``n_views=14`` (the mixed loader pads every sample to the largest rig).  H36M
only has 4 views, so this script loads data with ``WebBridgeCanonical17Dataset``,
pads to 14 views, and supplies an explicit ``view_mask`` so the padded views are
ignored by the model.

Usage
-----
    python scripts/eval_v25_true_gt_h36m_test.py \
        --checkpoint outputs/omniview_fusion_v25_h36m_true_gt_medium.pth \
        --s9 data/h36m_true_gt/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
        --s11 data/h36m_true_gt/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
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
from experiments.eval_omniview_fusion_v5_h36m import (
    build_model,
    load_checkpoint,
)


def collate_with_mask(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    dataset_ids = torch.tensor([b[5] for b in batch], dtype=torch.long)
    view_mask = torch.stack([b[6] for b in batch], dim=0)
    return x, y, K, R, t, dataset_ids, view_mask


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
        collate_fn=collate_with_mask,
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate v25 checkpoint on true-GT H36M test subjects S9/S11",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/omniview_fusion_v25_h36m_true_gt_medium.pth",
        help="Path to v25 .pth checkpoint",
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
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--out_json", type=str, default="outputs/eval_v25_true_gt_h36m_test.json")
    parser.add_argument("--config_json", type=str, default=None, help="Optional path to training config JSON")
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

    # build_model expects an argparse.Namespace with architecture flags.
    # Map a few config keys that differ between the trainer and the eval helper.
    model_args = Namespace(**config)
    model_args.entropy_weight = config.get("attention_entropy_weight", 0.01)
    model_args.adaptive_view_target_k = config.get("adaptive_view_k", 2)
    model_args.rotation_max_rot_deg = config.get("rotation_max_rot_deg", 2.0)
    model = build_model(model_args, n_views=n_views, j=j).to(device)
    load_checkpoint(model, args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

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
        results[name] = {
            "mpjpe_mm": float(report["mpjpe"]),
            "pa_mpjpe_mm": float(report["pa_mpjpe"]),
            "n_frames": preds.shape[0],
        }

    print("\n--- Summary ---")
    for name in ("S9", "S11"):
        print(f"{name}: MPJPE={results[name]['mpjpe_mm']:.2f}mm  PA-MPJPE={results[name]['pa_mpjpe_mm']:.2f}mm")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
