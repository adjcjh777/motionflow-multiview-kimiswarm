"""Compare DLT baseline vs. learnable Gauss-Newton triangulation on MPI-INF-3DHP.

Usage
-----
    conda run -n mf python experiments/compare_learned_tri_v1_mpiinf3dhp.py \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
        --dlt_ckpt outputs/ray_attention_temporal_residual_v2.pth \
        --gn_ckpt outputs/ray_attention_temporal_learned_tri_v1_smoke.pth \
        --clip_len 13
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_learned_tri_v1 import (
    RayAttentionFusionModelTemporalResidualLearnedTri,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
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
    parser = argparse.ArgumentParser(description="Compare DLT vs Gauss-Newton triangulation")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--dlt_ckpt", type=str, required=True, help="DLT baseline checkpoint")
    parser.add_argument("--gn_ckpt", type=str, required=True, help="Gauss-Newton checkpoint")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--gn_iters", type=int, default=3)
    parser.add_argument("--gn_damping", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = TemporalClipDataset(args.val, args.clip_len)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    sample = np.load(args.val)
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]

    # DLT baseline.
    dlt_model = RayAttentionFusionModelTemporalResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    dlt_model.load_state_dict(torch.load(args.dlt_ckpt, map_location=device), strict=False)
    dlt_err = evaluate(dlt_model, val_loader, device)
    print(f"DLT baseline MPJPE: {dlt_err*1000:.2f} mm")

    # Gauss-Newton model.
    gn_model = RayAttentionFusionModelTemporalResidualLearnedTri(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
        gn_iters=args.gn_iters,
        gn_damping=args.gn_damping,
    ).to(device)
    gn_model.load_state_dict(torch.load(args.gn_ckpt, map_location=device), strict=False)
    gn_err = evaluate(gn_model, val_loader, device)
    print(f"Gauss-Newton MPJPE: {gn_err*1000:.2f} mm")

    delta = (gn_err - dlt_err) * 1000
    print(f"Difference (GN - DLT): {delta:+.2f} mm")


if __name__ == "__main__":
    main()
