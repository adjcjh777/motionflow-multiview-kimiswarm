"""Train VisibilityGatedFusionModel on MPI-INF-3DHP clips.

Usage
-----
    conda run -n mf python experiments/train_visibility_gated_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --epochs 30

A built-in smoke test (no real data required) can be run with ``--smoke``.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_cameras
from motionflow_mv.data.temporal_clip_dataset import (
    RandomClipDataset,
    TemporalClipDataset,
    augment_clip,
    collate_fn,
    set_seed,
)
from motionflow_mv.fusion.visibility_gated_fusion import (
    VisibilityGatedFusionModel,
    _make_cameras,
)
from motionflow_mv.fusion.graph_joint_relation import MPI_INF_3DHP_28_PARENTS
from motionflow_mv.losses import bone_length_loss, reprojection_loss


def make_synthetic_npz(path: str, n_frames: int, n_views: int, j: int, seed: int = 0):
    """Create a tiny synthetic multi-view clip for smoke testing."""
    rng = np.random.RandomState(seed)
    cameras = _make_cameras(n_views)

    K = np.stack([cam.K for cam in cameras], axis=0).astype(np.float32)
    R = np.stack([cam.R for cam in cameras], axis=0).astype(np.float32)
    t = np.stack([cam.t for cam in cameras], axis=0).astype(np.float32)

    # Random 3D skeleton near the origin.
    base = rng.randn(j, 3) * 0.3
    joints_3d = base[None, :, :] + rng.randn(n_frames, j, 3) * 0.05
    joints_3d = joints_3d.astype(np.float32)

    # Project into each view.
    P = K @ np.concatenate([R, t[..., None]], axis=-1)  # (V, 3, 4)
    Xh = np.concatenate([joints_3d, np.ones((n_frames, j, 1), dtype=np.float32)], axis=-1)  # (T, J, 4)
    proj = np.einsum("vik,tjk->vtji", P, Xh)  # (V, T, J, 3)
    proj = proj.transpose(1, 0, 2, 3)  # (T, V, J, 3)
    z = proj[..., 2:]
    points_2d = proj[..., :2] / z

    confidences = np.ones((n_frames, n_views, j), dtype=np.float32)

    np.savez(
        path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=K,
        camera_R=R,
        camera_t=t,
    )


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(description="Train visibility-gated ray-attention fusion on MPI-INF-3DHP")
    parser.add_argument("--train", type=str, nargs="+", required=False, help="Train .npz files")
    parser.add_argument("--val", type=str, required=False, help="Validation .npz file")
    parser.add_argument("--smoke", action="store_true", help="Run a CPU-only smoke test on synthetic data")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--reproj_weight", type=float, default=0.0)
    parser.add_argument("--bone_weight", type=float, default=0.0)
    parser.add_argument("--occlusion_loss_weight", type=float, default=0.1, help="Weight for visibility BCE loss")
    parser.add_argument("--visibility_threshold", type=float, default=0.5)
    parser.add_argument("--min_visible_views", type=int, default=2)
    parser.add_argument("--cam_aug_rot", type=float, default=0.5)
    parser.add_argument("--cam_aug_trans", type=float, default=0.005)
    parser.add_argument("--cam_aug_focal", type=float, default=0.01)
    parser.add_argument("--cam_aug_pp", type=float, default=2.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/visibility_gated_mpiinf3dhp.pth")
    args = parser.parse_args()

    if args.smoke:
        smoke_dir = Path("tmp/visibility_gated_smoke")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_path = smoke_dir / "train.npz"
        val_path = smoke_dir / "val.npz"
        make_synthetic_npz(str(train_path), n_frames=120, n_views=4, j=17, seed=0)
        make_synthetic_npz(str(val_path), n_frames=40, n_views=4, j=17, seed=1)
        args.train = [str(train_path)]
        args.val = str(val_path)
        args.epochs = 2
        args.train_samples = 200
        args.batch_size = 2
        args.d = 32
        args.residual_hidden = 64
        args.clip_len = 9
        print(f"[smoke] using synthetic data: {train_path}, {val_path}")

    if not args.train or not args.val:
        parser.error("--train and --val are required unless --smoke is used")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_datasets = [RandomClipDataset(p, args.clip_len, n_samples=args.train_samples) for p in args.train]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.val_stride)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(
        f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, "
        f"residual_hidden={args.residual_hidden}"
    )

    model = VisibilityGatedFusionModel(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
        visibility_threshold=args.visibility_threshold,
        min_visible_views=args.min_visible_views,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
        print(f"Resumed from {args.resume}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            K, R, t = perturb_cameras(
                K,
                R,
                t,
                rot_std=args.cam_aug_rot,
                trans_std=args.cam_aug_trans,
                focal_std=args.cam_aug_focal,
                pp_std=args.cam_aug_pp,
            )

            optimizer.zero_grad()
            pred, _, v_logits = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)

            if args.occlusion_loss_weight > 0.0:
                visibility_target = (xb[..., 2] > 0).float()
                loss_occ = F.binary_cross_entropy_with_logits(v_logits, visibility_target)
                loss = loss + args.occlusion_loss_weight * loss_occ

            if args.reproj_weight > 0.0:
                points_2d = xb[..., :2]
                conf = xb[..., 2]
                loss_reproj = reprojection_loss(pred, points_2d, K, R, t, confidences=conf)
                loss = loss + args.reproj_weight * loss_reproj

            if args.bone_weight > 0.0:
                loss_bone = bone_length_loss(pred, yb, MPI_INF_3DHP_28_PARENTS)
                loss = loss + args.bone_weight * loss_bone

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
