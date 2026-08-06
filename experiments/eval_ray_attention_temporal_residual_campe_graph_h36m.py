"""Evaluate RayAttentionFusionModelTemporalResidualCamPEGraph on H36M."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_campe_graph_model import RayAttentionFusionModelTemporalResidualCamPEGraph


class TemporalClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, clip_len: int, stride: int = 1, gt_scale: float = 1.0, camera_scale: float = 1.0):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float() * gt_scale
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float() * camera_scale
        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gt_scale", type=float, default=1.0)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    dataset = TemporalClipDataset(args.dataset, args.clip_len, gt_scale=args.gt_scale, camera_scale=args.camera_scale)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = RayAttentionFusionModelTemporalResidualCamPEGraph(
        j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_temporal_layers
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    mpjpe = evaluate(model, loader, device)
    print(f"CamPE+GraphJR MPJPE: {mpjpe:.4f}m ({mpjpe*1000:.2f}mm)")


if __name__ == "__main__":
    main()
