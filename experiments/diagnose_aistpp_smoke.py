"""Diagnose the AIST++ smoke gap between v25/v80 and DLT.

Loads the saved v25/v80 checkpoints (trained on the AIST++-only smoke split)
and evaluates them on the same validation clip. Also computes the DLT baseline
on the raw canonical .npz for comparison.

Usage
-----
    python experiments/diagnose_aistpp_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.webbridge_mixed_dataset import (
    WebBridgeMixedDataset,
    webbridge_mixed_collate_fn,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt

# Import the model builder from the training script.
sys.path.insert(0, str(Path(__file__).parent))
from train_omniview_fusion_v5_webbridge_multi import (
    build_model_from_args,
    build_datasets,
)


def _build_P(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.zeros((K.shape[0], 3, 4), dtype=np.float64)
    for v in range(K.shape[0]):
        Rt = np.concatenate([R[v], t[v][:, None]], axis=1)
        P[v] = K[v] @ Rt
    return P


def dlt_mpjpe(npz_path: str) -> tuple[float, float]:
    """Return (direct MJE mm, root-aligned MPJPE mm) for a canonical .npz."""
    data = np.load(npz_path)
    p2d = data["points_2d"]
    j3d = data["joints_3d"]
    K = data["camera_K"]
    R = data["camera_R"]
    t = data["camera_t"]
    P = _build_P(K, R, t)
    n_frames, _, n_joints, _ = p2d.shape
    re_tri = np.zeros_like(j3d)
    for f in range(n_frames):
        for j in range(n_joints):
            re_tri[f, j] = triangulate_dlt(p2d[f, :, j], P)

    direct = float(np.linalg.norm(re_tri - j3d, axis=-1).mean() * 1000.0)
    pred_c = re_tri - re_tri.mean(axis=-2, keepdims=True)
    gt_c = j3d - j3d.mean(axis=-2, keepdims=True)
    root = float(np.linalg.norm(pred_c - gt_c, axis=-1).mean() * 1000.0)
    return direct, root


def load_args_from_config(config_path: Path) -> SimpleNamespace:
    with open(config_path) as f:
        cfg = json.load(f)
    args = SimpleNamespace(**cfg)
    # Provide fallbacks the model builder expects.
    for key, value in cfg.items():
        setattr(args, key, value)
    return args


def evaluate_checkpoint(checkpoint_path: Path, config_path: Path, device: torch.device, batch_size: int = 1):
    args = load_args_from_config(config_path)
    # Force CPU-only eval and single-worker data loading.
    args.num_workers = 0
    args.batch_size = batch_size

    # Rebuild the exact dataset used for the smoke run.
    train_dataset, val_dataset, n_views, n_joints = build_datasets(args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=webbridge_mixed_collate_fn,
        num_workers=0,
    )

    model = build_model_from_args(args, n_joints=n_joints, n_views=n_views, device=device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y, K, R, t, domain_id = batch
            x = x.to(device)
            y = y.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            domain_id = domain_id.to(device)
            out = model(x, K=K, R=R, t=t, domain_id=domain_id)
            pred = out[0]
            all_pred.append(pred.cpu())
            all_gt.append(y.cpu())

    pred = torch.cat(all_pred, dim=0)
    gt = torch.cat(all_gt, dim=0)
    direct = float((pred - gt).norm(dim=-1).mean().item() * 1000.0)
    pred_c = pred - pred.mean(dim=-2, keepdim=True)
    gt_c = gt - gt.mean(dim=-2, keepdim=True)
    root = float((pred_c - gt_c).norm(dim=-1).mean().item() * 1000.0)
    return direct, root, len(val_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="AIST++ smoke diagnostic")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=1, help="evaluation batch size")
    args_cli = parser.parse_args()
    device = torch.device(args_cli.device)
    bs = args_cli.batch_size

    val_path = "data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch03_multiview.npz"
    dlt_direct, dlt_root = dlt_mpjpe(val_path)
    print(f"DLT on {val_path}:")
    print(f"  direct MJE    = {dlt_direct:.2f} mm")
    print(f"  root MPJPE    = {dlt_root:.2f} mm")

    for name in ["v25", "v80"]:
        ckpt = Path(f"outputs/omniview_fusion_{name}_aist_only_smoke.pth")
        cfg = ckpt.with_suffix(".config.json")
        if not ckpt.exists():
            print(f"\n{name.upper()} checkpoint not found: {ckpt}")
            continue
        print(f"\nEvaluating {name.upper()} on AIST++ smoke val (batch_size={bs}) ...")
        try:
            direct, root, n = evaluate_checkpoint(ckpt, cfg, device, batch_size=bs)
            print(f"  {name} direct MJE = {direct:.2f} mm")
            print(f"  {name} root MPJPE  = {root:.2f} mm")
            print(f"  val clips        = {n}")
            final_ckpt = Path(str(ckpt).replace(".pth", "_final.pth"))
            if final_ckpt.exists() and final_ckpt != ckpt:
                d, r, _ = evaluate_checkpoint(final_ckpt, cfg, device, batch_size=bs)
                print(f"  {name}_final direct MJE = {d:.2f} mm")
                print(f"  {name}_final root MPJPE  = {r:.2f} mm")
        except Exception as exc:
            print(f"  ERROR evaluating {name}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
