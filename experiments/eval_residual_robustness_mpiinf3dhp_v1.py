"""Robustness evaluation for RayAttentionFusionModelTemporalResidual.

Perturbs the validation set with increasing levels of Gaussian 2D noise,
random joint occlusions, and random 2D outliers, then reports MPJPE,
PA-MPJPE, and PCK.  Results are saved as JSON, CSV, and PNG plots.

Usage
-----
    conda run -n mf python experiments/eval_residual_robustness_mpiinf3dhp_v1.py \
        --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --batch_size 8 \
        --out_dir outputs/robustness_residual_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a long canonical .npz sequence."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

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
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def perturb_noise(x, noise_std):
    """Add Gaussian pixel noise to (x, y) coordinates."""
    if noise_std <= 0:
        return x
    x = x.clone()
    x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    return x


def perturb_occlusion(x, occlusion_rate):
    """Zero out confidences for random (view, joint) pairs."""
    if occlusion_rate <= 0:
        return x
    x = x.clone()
    T, V, J = x.shape[1], x.shape[2], x.shape[3]
    mask = torch.rand(T, V, J, device=x.device) > occlusion_rate
    x[..., 2] = x[..., 2] * mask
    return x


def perturb_outliers(x, outlier_rate, outlier_scale=100.0):
    """Replace random (view, joint) 2D detections with large outliers."""
    if outlier_rate <= 0:
        return x
    x = x.clone()
    T, V, J = x.shape[1], x.shape[2], x.shape[3]
    outlier_mask = torch.rand(T, V, J, device=x.device) < outlier_rate
    outlier = (torch.rand(T, V, J, 2, device=x.device) - 0.5) * 2 * outlier_scale
    x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    # Also zero confidence so the model can learn to ignore them.
    x[..., 2] = x[..., 2] * (~outlier_mask).float()
    return x


def evaluate_perturbation(model, loader, device, perturb_fn, metrics_out):
    """Run one perturbation condition and compute metrics."""
    model.eval()
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb = perturb_fn(xb)
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)

            pred, _ = model(xb, K=K, R=R, t=t)

            all_pred.append(pred.cpu().numpy())
            all_gt.append(yb.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)  # (N_clips, T, J, 3)
    all_gt = np.concatenate(all_gt, axis=0)
    pred_flat = all_pred.reshape(-1, all_pred.shape[-2], 3)
    gt_flat = all_gt.reshape(-1, all_gt.shape[-2], 3)

    # Convert meters to millimeters.
    pred_mm = pred_flat * 1000.0
    gt_mm = gt_flat * 1000.0

    report = compute_all_metrics(pred_mm, gt_mm)
    metrics_out["mpjpe_mm"] = float(report["mpjpe"])
    metrics_out["pa_mpjpe_mm"] = float(report["pa_mpjpe"])
    metrics_out["pck_50mm"] = float(report.get("pck@50mm", None))
    metrics_out["pck_100mm"] = float(report.get("pck@100mm", None))
    metrics_out["pck_150mm"] = float(report.get("pck@150mm", None))
    metrics_out["pck_auc_150mm"] = float(report["pck_auc"])


def main():
    parser = argparse.ArgumentParser(
        description="Robustness evaluation for temporal residual ray-attention model"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1, help="Stride between validation clips (higher = fewer clips)")
    parser.add_argument("--out_dir", type=str, default="outputs/robustness_residual_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--noise_levels", type=float, nargs="+", default=[0, 2, 5, 10, 20],
        help="Gaussian noise std levels in pixels",
    )
    parser.add_argument(
        "--occlusion_rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5],
        help="Fraction of (view, joint) detections to occlude",
    )
    parser.add_argument(
        "--outlier_rates", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20],
        help="Fraction of (view, joint) detections to corrupt as outliers",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer dimensions from data.
    data = np.load(args.val)
    n_views = int(data["camera_K"].shape[0])
    j = int(data["points_2d"].shape[2])
    print(f"Data: {args.val}")
    print(f"n_views={n_views}, n_joints={j}, clip_len={args.clip_len}, d={args.d}")

    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.stride)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load model.
    model = RayAttentionFusionModelTemporalResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    results = {
        "checkpoint": str(args.checkpoint),
        "val_dataset": str(args.val),
        "n_views": int(n_views),
        "n_joints": int(j),
        "clip_len": int(args.clip_len),
        "d": int(args.d),
        "n_temporal_layers": int(args.n_temporal_layers),
        "residual_hidden": int(args.residual_hidden),
        "noise": [],
        "occlusion": [],
        "outliers": [],
    }

    # Noise sweep.
    print("\n=== Gaussian noise sweep ===")
    for noise_std in args.noise_levels:
        entry = {"noise_std_px": float(noise_std)}
        evaluate_perturbation(
            model, val_loader, device,
            lambda x: perturb_noise(x, noise_std),
            entry,
        )
        results["noise"].append(entry)
        print(f"  noise {noise_std:.1f}px -> MPJPE {entry['mpjpe_mm']:.2f} mm")

    # Occlusion sweep.
    print("\n=== Occlusion sweep ===")
    for occ_rate in args.occlusion_rates:
        entry = {"occlusion_rate": float(occ_rate)}
        evaluate_perturbation(
            model, val_loader, device,
            lambda x: perturb_occlusion(x, occ_rate),
            entry,
        )
        results["occlusion"].append(entry)
        print(f"  occlusion {occ_rate:.2f} -> MPJPE {entry['mpjpe_mm']:.2f} mm")

    # Outlier sweep.
    print("\n=== Outlier sweep ===")
    for outlier_rate in args.outlier_rates:
        entry = {"outlier_rate": float(outlier_rate)}
        evaluate_perturbation(
            model, val_loader, device,
            lambda x: perturb_outliers(x, outlier_rate),
            entry,
        )
        results["outliers"].append(entry)
        print(f"  outlier {outlier_rate:.2f} -> MPJPE {entry['mpjpe_mm']:.2f} mm")

    # Save JSON.
    json_path = out_dir / "robustness_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON report to {json_path}")

    # Save CSV for easy table generation.
    csv_path = out_dir / "robustness_report.csv"
    with open(csv_path, "w") as f:
        f.write("perturbation,level,mpjpe_mm,pa_mpjpe_mm,pck_50mm,pck_100mm,pck_150mm,pck_auc_150mm\n")
        for key in ["noise", "occlusion", "outliers"]:
            for entry in results[key]:
                level_key = [k for k in entry if k != "mpjpe_mm" and "_mm" not in k and "pck" not in k][0]
                f.write(
                    f"{key},{entry[level_key]},{entry['mpjpe_mm']:.4f},{entry['pa_mpjpe_mm']:.4f},"
                    f"{entry['pck_50mm']:.4f},{entry['pck_100mm']:.4f},{entry['pck_150mm']:.4f},"
                    f"{entry['pck_auc_150mm']:.4f}\n"
                )
    print(f"Saved CSV report to {csv_path}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Noise.
        noise_levels = [e["noise_std_px"] for e in results["noise"]]
        noise_mpjpe = [e["mpjpe_mm"] for e in results["noise"]]
        axes[0].plot(noise_levels, noise_mpjpe, marker="o")
        axes[0].set_xlabel("Gaussian noise std (px)")
        axes[0].set_ylabel("MPJPE (mm)")
        axes[0].set_title("Noise Robustness")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Occlusion.
        occ_rates = [e["occlusion_rate"] for e in results["occlusion"]]
        occ_mpjpe = [e["mpjpe_mm"] for e in results["occlusion"]]
        axes[1].plot(occ_rates, occ_mpjpe, marker="s", color="orange")
        axes[1].set_xlabel("Occlusion rate")
        axes[1].set_ylabel("MPJPE (mm)")
        axes[1].set_title("Occlusion Robustness")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].yaxis.get_major_formatter().set_useOffset(False)

        # Outliers.
        out_rates = [e["outlier_rate"] for e in results["outliers"]]
        out_mpjpe = [e["mpjpe_mm"] for e in results["outliers"]]
        axes[2].plot(out_rates, out_mpjpe, marker="^", color="green")
        axes[2].set_xlabel("Outlier rate")
        axes[2].set_ylabel("MPJPE (mm)")
        axes[2].set_title("Outlier Robustness")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = out_dir / "robustness_plots.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plots to {plot_path}")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
