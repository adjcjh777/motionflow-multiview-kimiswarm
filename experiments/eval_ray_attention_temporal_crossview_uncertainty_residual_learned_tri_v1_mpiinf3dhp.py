"""Evaluate a trained RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1 on MPI-INF-3DHP clips.

Usage
-----
    conda run -n mf python experiments/eval_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.pth \
        --clip_len 13 --d 64 --n_st_layers 2 --batch_size 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model import RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1
from motionflow_mv.fusion.ray_attention_temporal_uncertainty_residual_learned_tri_v1_model import RayAttentionFusionModelTemporalUncertaintyResidualLearnedTriV1


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a canonical .npz sequence."""

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
            pred, _, _, _ = model(xb, K=K, R=R, t=t)
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
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_type", type=str, default="crossview", choices=["crossview", "temporal"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    dataset = TemporalClipDataset(args.dataset, args.clip_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    if args.model_type == "crossview":
        model = RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers
        ).to(device)
    else:
        model = RayAttentionFusionModelTemporalUncertaintyResidualLearnedTriV1(
            j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_st_layers
        ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    mpjpe = evaluate(model, loader, device)
    print(f"AdvancedV1 MPJPE: {mpjpe:.4f}m ({mpjpe*1000:.2f}mm)")


if __name__ == "__main__":
    main()
