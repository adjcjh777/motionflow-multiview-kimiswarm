"""Evaluate mixed-dataset principal-point correction model on MPI-INF-3DHP.

Mirrors ``eval_principal_point_model_mpiinf3dhp.py`` but dispatches through the
mixed-dataset model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.mixed_dataset import DATASET_REGISTRY
from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.ray_attention_temporal_mixed_residual_principal_point_model import RayAttentionFusionModelTemporalMixedResidualPrincipalPoint


class TemporalClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.camera_K = torch.from_numpy(data["camera_K"]).float()
        self.camera_R = torch.from_numpy(data["camera_R"]).float()
        self.camera_t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.camera_K, self.camera_R, self.camera_t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def evaluate_clean(model, loader, device, max_joints):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            dataset_ids = torch.zeros(xb.size(0), dtype=torch.long, device=device)
            pred, mask = model(xb, K=K, R=R, t=t, dataset_ids=dataset_ids)
            pred = pred[:, :, :max_joints]
            yb = yb[:, :, :max_joints]
            mask = mask[:, :, :max_joints]
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, max_joints, 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, max_joints, 3) * 1000.0
    return compute_all_metrics(preds, gts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=64)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--out_json", type=str, default="outputs/mixed_pp_eval.json")
    parser.add_argument("--focal_max_scale", type=float, default=0.0, help="Maximum predicted focal-length scale; 0 disables focal correction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    max_joints = DATASET_REGISTRY[args.val_dataset]["n_joints"]

    model = RayAttentionFusionModelTemporalMixedResidualPrincipalPoint(
        d=args.d,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    print("Clean evaluation...")
    clean_report = evaluate_clean(model, loader, device, max_joints)
    clean_summary = {k: float(v) for k, v in clean_report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}
    print(f"Clean: MPJPE={clean_summary['mpjpe']:.2f}mm PA={clean_summary['pa_mpjpe']:.2f}mm")

    out = {"clean": clean_summary}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved summary to {args.out_json}")


if __name__ == "__main__":
    main()
