"""Train ray-aware attention fusion v3 on multi-view data (e.g. H36M).

Adds optional bone-length (supervised L1) and skeleton-consistency
(temporal + symmetry) losses via experiments/train_utils.py.

Swarm-iter5 wiring note:
- CLI args --bone_weight, --consistency_weight, --skeleton_layout added.
- H36M/COCO/SMPL 17-joint topology presets live in train_utils.py.
- Verified: imports, py_compile, and loss sanity checks all pass.

Usage:
    python experiments/train_ray_attention_v3_h36m.py \
        --dataset data/h36m_hf/s_01_acts_02_..._16_multiview.npz \
        --epochs 50 --lr 1e-3 --d 64 --batch_size 32 \
        --bone_weight 0.1 --consistency_weight 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3
from experiments.train_utils import (
    bone_length_loss,
    skeleton_consistency_loss,
    H36M_17_PARENTS,
)


class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, idx: np.ndarray):
        self.x = torch.from_numpy(data["points_2d"][idx]).float()
        self.conf = torch.from_numpy(data["confidences"][idx]).float()
        self.y = torch.from_numpy(data["joints_3d"][idx]).float()
        # Cameras may be a single rig (V, ...) broadcast to all samples.
        if data["camera_K"].shape[0] == self.x.shape[0]:
            self.K = torch.from_numpy(data["camera_K"][idx]).float()
            self.R = torch.from_numpy(data["camera_R"][idx]).float()
            self.t = torch.from_numpy(data["camera_t"][idx]).float()
        else:
            self.K = torch.from_numpy(data["camera_K"]).float().unsqueeze(0).repeat(len(idx), 1, 1, 1)
            self.R = torch.from_numpy(data["camera_R"]).float().unsqueeze(0).repeat(len(idx), 1, 1, 1)
            self.t = torch.from_numpy(data["camera_t"]).float().unsqueeze(0).repeat(len(idx), 1, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.cat([self.x[idx], self.conf[idx, ..., None]], dim=-1)
        return x, self.y[idx], self.K[idx], self.R[idx], self.t[idx]


def collate_fn(batch):
    xb = torch.stack([b[0] for b in batch], dim=0)
    yb = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return xb, yb, K, R, t


def augment_batch(x, noise_std=0.5, dropout_rate=0.1, outlier_rate=0.02, outlier_scale=100.0):
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="outputs/synthetic_multiview_dataset.npz")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--bone_weight", type=float, default=0.1,
                        help="Weight for supervised bone-length L1 loss.")
    parser.add_argument("--consistency_weight", type=float, default=0.05,
                        help="Weight for skeleton consistency loss (temporal + symmetry).")
    parser.add_argument("--skeleton_layout", type=str, default="h36m17",
                        choices=["h36m17", "coco17", "smpl17"],
                        help="Skeleton topology preset for bone losses.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.dataset)
    n = data["joints_3d"].shape[0]
    n_val = int(n * args.val_ratio)
    perm = np.random.permutation(n)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_dataset = CameraDataset(data, train_idx)
    val_dataset = CameraDataset(data, val_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, K_shape={data['camera_K'].shape}")

    if args.skeleton_layout == "h36m17":
        skeleton_parents = H36M_17_PARENTS
    elif args.skeleton_layout == "coco17":
        from experiments.train_utils import COCO_17_PARENTS
        skeleton_parents = COCO_17_PARENTS
    elif args.skeleton_layout == "smpl17":
        from experiments.train_utils import SMPL17_PARENTS
        skeleton_parents = SMPL17_PARENTS

    # Symmetric bone indices for H36M 17-joint subset.
    # Left/right pairs: hip-knee, knee-ankle, shoulder-elbow, elbow-wrist.
    if args.skeleton_layout == "h36m17":
        left_bones = [4, 5, 10, 11]   # lhip-lknee, lknee-lankle, lshoulder-lelbow, lelbow-lwrist
        right_bones = [1, 2, 13, 14]  # rhip-rknee, rknee-rankle, rshoulder-relbow, relbow-rwrist
    else:
        left_bones, right_bones = None, None

    model = RayAttentionFusionModelV3(j=j, d=args.d, n_views=n_views).to(device)
    print(f"fusion_mlp weight shape: {model.fusion_mlp[0].weight.shape}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            if args.bone_weight > 0.0:
                loss = loss + bone_length_loss(pred, yb, parents=skeleton_parents, weight=args.bone_weight)
            if args.consistency_weight > 0.0:
                loss = loss + skeleton_consistency_loss(
                    pred, parents=skeleton_parents,
                    left_bones=left_bones, right_bones=right_bones,
                    temporal_weight=args.consistency_weight,
                    symmetry_weight=args.consistency_weight,
                )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        with torch.no_grad():
            for xb, yb, K, R, t in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                pred, _ = model(xb, K=K, R=R, t=t)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_mpjpe += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_mpjpe /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            dataset_stem = Path(args.dataset).stem
            torch.save(model.state_dict(), output_dir / f"ray_attention_v2_{dataset_stem}.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}m")

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / 'ray_attention_v2_synthetic.pth'}")


if __name__ == "__main__":
    main()
