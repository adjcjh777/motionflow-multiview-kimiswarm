"""Evaluate RayAttentionFusionModelTemporalCrossviewResidual on MPI-INF-3DHP.

Usage
-----
    conda run -n mf python experiments/eval_ray_attention_temporal_crossview_residual_mpiinf3dhp.py \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/crossview_residual_d128_h256_nst3_full.pth \
        --clip_len 13 --d 128 --n_st_layers 3 --residual_hidden 256
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidual,
)
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe, pck, pck_auc


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
    preds = []
    gts = []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    pred = np.concatenate(preds, axis=0)  # (N, T, J, 3)
    gt = np.concatenate(gts, axis=0)
    # Drop temporal dimension for per-frame metrics.
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])
    return pred, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_joint_layers", type=int, default=1)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    dataset = TemporalClipDataset(args.dataset, args.clip_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model = RayAttentionFusionModelTemporalCrossviewResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_joint_layers=args.n_joint_layers,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    pred, gt = evaluate(model, loader, device)
    # Infer unit: if GT is in meters (typical for _m.npz), scale to millimetres.
    scale = 1000.0 if gt.max() < 10.0 else 1.0
    err = mpjpe(pred, gt) * scale
    pa_err = pa_mpjpe(pred, gt) * scale
    pck_50 = pck(pred, gt, 50.0 / scale)
    pck_100 = pck(pred, gt, 100.0 / scale)
    pck_150 = pck(pred, gt, 150.0 / scale)
    auc, _, _ = pck_auc(pred, gt, max_threshold=150.0 / scale)

    print(f"MPJPE:     {err:.4f} mm")
    print(f"PA-MPJPE:  {pa_err:.4f} mm")
    print(f"PCK@50mm:  {pck_50:.4f}")
    print(f"PCK@100mm: {pck_100:.4f}")
    print(f"PCK@150mm: {pck_150:.4f}")
    print(f"AUC:       {auc:.4f}")


if __name__ == "__main__":
    main()
