"""Train ray-aware attention fusion on real Shelf/Campus GT.

Example:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_ray_attention_real.py \
        --data_root data/shelf_campus/Shelf_Seq1 --dataset_name shelf \
        --epochs 100 --batch_size 32 --d 64 --input_scale 0.01
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.shelf_loader import build_shelf_dataset
from motionflow_mv.fusion.ray_attention_model import RayAttentionFusionModel
from motionflow_mv.fusion.triangulation import triangulate_dlt


def build_projection_matrices(K, R, t):
    """Build (V, 3, 4) projection matrices from numpy arrays."""
    Rt = np.concatenate([R, t[..., None]], axis=-1)  # (V, 3, 4)
    return K @ Rt  # (V, 3, 4)


def dlt_baseline(points_2d, confidences, K_orig, R_orig, t_m):
    """Compute confidence-weighted DLT triangulation.

    Args:
        points_2d: (B, V, J, 2) image coordinates (already unscaled, in original units).
        confidences: (B, V, J)
        K_orig: (V, 3, 3) intrinsics.
        R_orig: (V, 3, 3) rotations.
        t_m: (V, 3) translation in meters.

    Returns:
        X: (B, J, 3) in meters.
    """
    P = build_projection_matrices(K_orig, R_orig, t_m)
    B, V, J, _ = points_2d.shape
    X = np.zeros((B, J, 3), dtype=np.float64)
    for b in range(B):
        for j in range(J):
            w = confidences[b, :, j]
            if w.sum() == 0:
                w = np.ones_like(w)
            X[b, j] = triangulate_dlt(points_2d[b, :, j], P, weights=w)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to Shelf_Seq1 or Campus_Seq1")
    parser.add_argument("--dataset_name", type=str, default="shelf", help="Dataset identifier for output names")
    parser.add_argument("--person_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--input_scale", type=float, default=1.0, help="Scale applied to intrinsics ONLY (2D points must stay in original units)")
    parser.add_argument("--noise_std", type=float, default=50.0, help="Gaussian 2D noise std for training augmentation")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Random per-view dropout probability during training")
    parser.add_argument("--outlier_rate", type=float, default=0.05, help="Fraction of (view,joint) observations to replace by outliers")
    parser.add_argument("--outlier_scale", type=float, default=5000.0, help="Magnitude of injected 2D outliers")
    parser.add_argument("--reproj_weight", type=float, default=0.0, help="Weight of the optional reprojection loss (0 to disable)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    points_2d, confidences, joints_3d, cameras = build_shelf_dataset(Path(args.data_root), person_id=args.person_id)
    n_frames, n_views, n_joints, _ = points_2d.shape
    print(f"Loaded {args.data_root}: T={n_frames}, V={n_views}, J={n_joints}")

    # Convert to meters (camera t and 3D GT).
    joints_3d = joints_3d / 100.0
    for cam in cameras:
        cam.t = cam.t / 100.0

    # Store unscaled camera parameters for the DLT baseline.
    K_orig = np.stack([cam.K for cam in cameras])
    R_orig = np.stack([cam.R for cam in cameras])
    t_m = np.stack([cam.t for cam in cameras])

    # Optionally scale intrinsics to keep model inputs in a familiar numeric range.
    # 2D points must remain in the same units as the intrinsics, so we only scale K.
    for cam in cameras:
        cam.K = cam.K * args.input_scale

    # Random split (temporal independence is not required here).
    perm = np.random.permutation(n_frames)
    n_val = int(n_frames * args.val_ratio)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    class ShelfDataset(torch.utils.data.Dataset):
        def __init__(self, p2d, conf, target):
            self.x = torch.from_numpy(p2d).float()
            self.conf = torch.from_numpy(conf).float()
            self.y = torch.from_numpy(target).float()

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            x = torch.cat([self.x[idx], self.conf[idx, ..., None]], dim=-1)
            return x, self.y[idx]

    train_ds = ShelfDataset(points_2d[train_idx], confidences[train_idx], joints_3d[train_idx])
    val_ds = ShelfDataset(points_2d[val_idx], confidences[val_idx], joints_3d[val_idx])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

    # Prepare per-sample camera tensors (single rig, will broadcast).
    K = torch.from_numpy(np.stack([cam.K for cam in cameras])).float().unsqueeze(0)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras])).float().unsqueeze(0)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras])).float().unsqueeze(0)

    # Original projection matrices for the optional reprojection loss.
    P_orig = build_projection_matrices(K_orig, R_orig, t_m)  # (V, 3, 4)
    P_orig = torch.from_numpy(P_orig).float().to(device)

    model = RayAttentionFusionModel(j=n_joints, d=args.d, n_views=n_views).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    def reprojection_loss(pred_3d, points_2d, confidences):
        """Compute weighted L2 reprojection error of pred_3d onto points_2d.

        Args:
            pred_3d: (B, J, 3) world coordinates
            points_2d: (B, V, J, 2) image coordinates
            confidences: (B, V, J) non-negative weights
        Returns:
            loss: scalar
        """
        B, J, _ = pred_3d.shape
        V = points_2d.shape[1]
        X_h = torch.cat([pred_3d, torch.ones(B, J, 1, device=pred_3d.device)], dim=-1)  # (B, J, 4)
        # proj_h: (V, 3, 4) @ (B, J, 4) -> (B, V, J, 3)
        proj_h = torch.einsum('vpq,bjq->bvjp', P_orig, X_h)
        proj_2d = proj_h[..., :2] / (proj_h[..., 2:3] + 1e-8)
        diff = proj_2d - points_2d  # (B, V, J, 2)
        w = confidences / (confidences.sum(dim=(1, 2), keepdim=True) + 1e-8)
        loss = (w[..., None] * diff ** 2).sum() / (B * V)
        return loss

    def augment_batch(x):
        """Add 2D noise, random view dropout, and sparse outliers to the input tensor."""
        B, V, J, _ = x.shape
        if args.noise_std > 0:
            x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * args.noise_std
        if args.dropout_rate > 0:
            mask = (torch.rand(B, V, J, device=x.device) > args.dropout_rate).float()
            x[..., 2] = x[..., 2] * mask
        if args.outlier_rate > 0:
            outlier_mask = torch.rand(B, V, J, device=x.device) < args.outlier_rate
            # Replace selected observations with a large random offset
            outlier = (torch.rand(B, V, J, 2, device=x.device) - 0.5) * 2 * args.outlier_scale
            x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
        return x

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    best_val = float("inf")
    best_mpjpe = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_clean = xb.clone()
            xb = augment_batch(xb)
            Kb, Rb, tb = K.to(device), R.to(device), t.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb, K=Kb, R=Rb, t=tb)
            loss = criterion(pred, yb)
            if args.reproj_weight > 0:
                loss = loss + args.reproj_weight * reprojection_loss(pred, xb_clean[..., :2], xb_clean[..., 2])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb, K=K.to(device), R=R.to(device), t=t.to(device))
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_mpjpe += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_mpjpe /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            best_mpjpe = val_mpjpe
            ckpt_path = output_dir / f"ray_attention_{args.dataset_name}.pth"
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}m")

    print(f"Best val_loss={best_val:.6f}, best val_MPJPE={best_mpjpe:.4f}m, checkpoint: {ckpt_path}")

    # DLT baseline on validation set (use original 2D and intrinsics).
    val_p2d = points_2d[val_idx]
    val_conf = confidences[val_idx]
    val_y = joints_3d[val_idx]
    dlt_X = dlt_baseline(val_p2d, val_conf, K_orig, R_orig, t_m)
    dlt_mpjpe = np.linalg.norm(dlt_X - val_y, axis=-1).mean()
    print(f"DLT baseline val_MPJPE={dlt_mpjpe:.4f}m")


if __name__ == "__main__":
    main()
