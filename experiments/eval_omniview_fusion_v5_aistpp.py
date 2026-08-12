"""Evaluate OmniMultiViewFusionV5 on the AIST++ test split.

Reports per-clip and aggregate MPJPE/PA-MPJPE over all .npz files listed in
the ``test`` section of a YAML split manifest.

Usage
-----
    python experiments/eval_omniview_fusion_v5_aistpp.py \
        --checkpoint outputs/ablations/v25_aistpp_full_medium_a800.pth \
        --split configs/splits/aistpp_train_val_test.yaml

The script loads the saved training config (``<checkpoint>.config.json``),
reconstructs the exact architecture, loads the checkpoint, and runs inference
on every test .npz file.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TemporalClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


# ---------------------------------------------------------------------------
# Model construction / checkpoint loading
# ---------------------------------------------------------------------------

def _load_config(checkpoint_path: str) -> Dict[str, Any]:
    config_path = Path(checkpoint_path).with_suffix(".config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def _build_model_from_config(config: Dict[str, Any], n_views: int, n_joints: int) -> OmniMultiViewFusionV5:
    sig = inspect.signature(OmniMultiViewFusionV5.__init__)
    kwargs: Dict[str, Any] = {"j": n_joints, "n_views": n_views}
    for name in sig.parameters:
        if name in ("self", "j", "n_views"):
            continue
        if name in config:
            kwargs[name] = config[name]

    # argparse name -> constructor name mapping.
    if "entropy_weight" not in kwargs and "attention_entropy_weight" in config:
        kwargs["entropy_weight"] = config["attention_entropy_weight"]

    return OmniMultiViewFusionV5(**kwargs)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected[:5]}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_file(
    model: torch.nn.Module,
    npz_path: str,
    clip_len: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    dataset = TemporalClipDataset(npz_path, clip_len, stride=stride)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model.eval()
    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb = xb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            yb = yb.to(device)
            pred = model(xb, K=K, R=R, t=t)[0]
            all_preds.append(pred.cpu().numpy())
            all_gts.append(yb.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).reshape(-1, all_preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(all_gts, axis=0).reshape(-1, all_gts[0].shape[-2], 3) * 1000.0
    report = compute_all_metrics(preds, gts)

    return {
        "mpjpe_mm": float(report["mpjpe"]),
        "pa_mpjpe_mm": float(report["pa_mpjpe"]),
        "n_frames": int(preds.shape[0]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OmniMultiViewFusionV5 on AIST++ test split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--split", type=str, required=True, help="YAML split file with a 'test' list")
    parser.add_argument("--clip_len", type=int, default=13, help="Temporal clip length")
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.split, "r") as f:
        split_cfg = yaml.safe_load(f)

    test_files = split_cfg.get("test", [])
    if not test_files:
        raise ValueError(f"No test files found in {args.split}")

    config = _load_config(args.checkpoint)

    # Infer number of views / joints from the first test file.
    sample = np.load(test_files[0])
    n_views = int(sample["camera_K"].shape[0])
    n_joints = int(sample["points_2d"].shape[2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; checkpoint: {args.checkpoint}")
    print(f"Test files: {len(test_files)}; n_views={n_views}; n_joints={n_joints}")

    model = _build_model_from_config(config, n_views=n_views, n_joints=n_joints).to(device)
    _load_checkpoint(model, args.checkpoint)

    per_file_results: List[Dict[str, Any]] = []
    mpjpe_values: List[float] = []
    pa_mpjpe_values: List[float] = []

    for npz_path in test_files:
        print(f"Evaluating {npz_path}...")
        result = evaluate_file(
            model,
            npz_path,
            clip_len=args.clip_len,
            stride=args.val_stride,
            batch_size=args.batch_size,
            device=device,
        )
        result["npz"] = npz_path
        per_file_results.append(result)
        mpjpe_values.append(result["mpjpe_mm"])
        pa_mpjpe_values.append(result["pa_mpjpe_mm"])
        print(f"  MPJPE={result['mpjpe_mm']:.2f}mm  PA-MPJPE={result['pa_mpjpe_mm']:.2f}mm")

    aggregate = {
        "mean_mpjpe_mm": float(np.mean(mpjpe_values)),
        "mean_pa_mpjpe_mm": float(np.mean(pa_mpjpe_values)),
        "std_mpjpe_mm": float(np.std(mpjpe_values)),
        "std_pa_mpjpe_mm": float(np.std(pa_mpjpe_values)),
        "min_mpjpe_mm": float(np.min(mpjpe_values)),
        "max_mpjpe_mm": float(np.max(mpjpe_values)),
        "n_clips": len(test_files),
    }

    print("\nAggregate test-set results:")
    print(f"  Mean MPJPE     = {aggregate['mean_mpjpe_mm']:.2f} mm")
    print(f"  Mean PA-MPJPE  = {aggregate['mean_pa_mpjpe_mm']:.2f} mm")
    print(f"  Std MPJPE      = {aggregate['std_mpjpe_mm']:.2f} mm")
    print(f"  Std PA-MPJPE   = {aggregate['std_pa_mpjpe_mm']:.2f} mm")

    out_json = Path(args.out_json) if args.out_json else Path(args.checkpoint).with_suffix(".aistpp_test.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "aggregate": aggregate,
                "per_file": per_file_results,
            },
            f,
            indent=2,
        )
    print(f"Saved results -> {out_json}")


if __name__ == "__main__":
    main()
