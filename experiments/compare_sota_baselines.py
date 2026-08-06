"""Compare the best learned model against geometric/learning baselines on MPI-INF-3DHP.

Methods
-------
- Confidence-weighted DLT
- Robust IRLS triangulation
- Iskakov-style learned triangulation
- Best learned model (e.g. PP baseline)

Usage
-----
    python experiments/compare_sota_baselines.py \
        --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --out_json outputs/sota_comparison_mpi.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
from motionflow_mv.fusion.robust_triangulation_baseline import triangulate_irls
from motionflow_mv.fusion.triangulation import triangulate_dlt


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


def _projection_matrices(K, R, t):
    """Build (V, 3, 4) projection matrices from K, R, t."""
    # K, R, t are numpy arrays on CPU.
    Rt = np.concatenate([R, t[..., None]], axis=-1)  # (V, 3, 4)
    P = K @ Rt
    return P


def dlt_baseline(points_2d, confidences, K, R, t):
    """Confidence-weighted DLT baseline."""
    # points_2d: (B, T, V, J, 2), confidences: (B, T, V, J)
    B, T, V, J, _ = points_2d.shape
    preds = []
    for b in range(B):
        preds_b = []
        for tt in range(T):
            pred_j = []
            for j in range(J):
                p2d = points_2d[b, tt, :, j, :].cpu().numpy()
                conf = confidences[b, tt, :, j].cpu().numpy()
                K_np = K[b].cpu().numpy()
                R_np = R[b].cpu().numpy()
                t_np = t[b].cpu().numpy()
                P = _projection_matrices(K_np, R_np, t_np)
                pred = triangulate_dlt(p2d, P, weights=conf)
                pred_j.append(pred)
            preds_b.append(np.stack(pred_j, axis=0))
        preds.append(np.stack(preds_b, axis=0))
    return np.stack(preds, axis=0)  # (B, T, J, 3)


def irls_baseline(points_2d, confidences, K, R, t):
    """Robust IRLS triangulation baseline."""
    B, T, V, J, _ = points_2d.shape
    preds = []
    for b in range(B):
        preds_b = []
        for tt in range(T):
            pred_j = []
            for j in range(J):
                p2d = points_2d[b, tt, :, j, :].cpu().numpy()
                conf = confidences[b, tt, :, j].cpu().numpy()
                K_np = K[b].cpu().numpy()
                R_np = R[b].cpu().numpy()
                t_np = t[b].cpu().numpy()
                P = _projection_matrices(K_np, R_np, t_np)
                pred = triangulate_irls(p2d, P, confidences=conf)
                pred_j.append(pred)
            preds_b.append(np.stack(pred_j, axis=0))
        preds.append(np.stack(preds_b, axis=0))
    return np.stack(preds, axis=0)


def evaluate_baseline(baseline_fn, loader, device):
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb = xb.to(device)
            pred = baseline_fn(xb[..., :2], xb[..., 2], K, R, t)
            preds.append(pred)
            gts.append(yb.numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def main():
    parser = argparse.ArgumentParser(description="SOTA baseline comparison for MotionFlow-MultiView")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="outputs/sota_comparison_mpi.json")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Learned model
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    learned_report = compute_all_metrics(preds, gts)

    # Baselines
    dlt_report = evaluate_baseline(dlt_baseline, loader, device)
    irls_report = evaluate_baseline(irls_baseline, loader, device)

    def summarize(report):
        return {k: float(v) for k, v in report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}

    results = {
        "dlt": summarize(dlt_report),
        "irls": summarize(irls_report),
        "learned_pp": summarize(learned_report),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    for name, r in results.items():
        print(f"{name:20s} MPJPE={r['mpjpe']:.2f}mm PA={r['pa_mpjpe']:.2f}mm PCK@50={r['pck@50mm']:.3f} AUC={r['pck_auc']:.3f}")


if __name__ == "__main__":
    main()
