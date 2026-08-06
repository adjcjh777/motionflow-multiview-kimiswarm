"""Diagnose why the trained PP correction head fails under real principal-point perturbation.

Usage
-----
    python experiments/diagnose_pp_correction.py \
        --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_intrinsics_with_delta
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        return_pp_delta=True,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    for cxcy_err in [0.0, 3.0, 5.0, 10.0]:
        deltas, true_offsets, mpjpe_list = [], [], []
        with torch.no_grad():
            for xb, yb, K, R, t in loader:
                xb = xb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                K_pert, true_pp_delta, _ = perturb_intrinsics_with_delta(K, focal_std=0.0, pp_std=cxcy_err)
                pred, _, pred_pp_delta = model(xb, K=K_pert, R=R, t=t)
                # pred_pp_delta shape: (B*T, V, 2)
                deltas.append(pred_pp_delta.cpu().numpy())
                true_offsets.append(-true_pp_delta.cpu().numpy())
                err = (pred - yb.to(device)).norm(dim=-1).mean() * 1000.0
                mpjpe_list.append(err.item())
        deltas = np.concatenate(deltas, axis=0)  # (N, V, 2)
        true_offsets = np.concatenate(true_offsets, axis=0)  # (N, V, 2)
        print(f"cxcy_err={cxcy_err:.1f}px  "
              f"pred_delta_mean=({deltas[...,0].mean():.3f}, {deltas[...,1].mean():.3f})  "
              f"true_offset_mean=({true_offsets[...,0].mean():.3f}, {true_offsets[...,1].mean():.3f})  "
              f"pred_delta_std=({deltas[...,0].std():.3f}, {deltas[...,1].std():.3f})  "
              f"MPJPE={np.mean(mpjpe_list):.2f}mm")


if __name__ == "__main__":
    main()
