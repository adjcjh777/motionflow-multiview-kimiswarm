"""Evaluate outputs/ray_attention_temporal_residual_full30.pth on MPI-INF-3DHP S2 Seq1."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


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
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end],
             self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def evaluate(checkpoint_path: str, val_path: str, clip_len: int = 13, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = TemporalClipDataset(val_path, clip_len)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    sample = np.load(val_path)
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]

    model = RayAttentionFusionModelTemporalResidual(
        j=j, d=64, n_views=n_views, n_temporal_layers=2, residual_hidden=128,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    mpjpe = total_err / total_count
    return mpjpe * 1000.0  # mm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/ray_attention_temporal_residual_full30.pth")
    parser.add_argument("--val", type=str, default="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    mpjpe = evaluate(args.checkpoint, args.val, args.clip_len, args.batch_size)
    print(f"MPJPE: {mpjpe:.2f} mm")
