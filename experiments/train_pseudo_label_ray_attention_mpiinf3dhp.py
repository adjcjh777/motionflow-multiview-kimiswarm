"""Semi-supervised teacher-student training with pseudo-labels for MPI-INF-3DHP.

This script extends the supervised ``train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py``
loop with a semi-supervised stage:

1. Train (or load) a teacher model on the labeled training set.
2. Use the teacher to generate 3-D pseudo-labels for an unlabeled dataset that has
   the same multi-view 2-D format but no ``joints_3d`` field.
3. Train a student model on a mixture of labeled and pseudo-labeled clips.
   Pseudo-labels are weighted by a confidence mask derived from the teacher's
   mean reprojection error, so unreliable teacher predictions are down-weighted.

Expected outcome
----------------
The student can effectively leverage cheap unlabeled multi-view video to
improve generalisation beyond the small labeled split, potentially pushing
MPJPE on MPI-INF-3DHP / WebBridge below the current anchor.

Usage
-----
    # 1. Generate pseudo-labels from a pre-trained teacher checkpoint.
    python experiments/train_pseudo_label_ray_attention_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --unlabeled data/webbridge/mpi_inf_3dhp/s_04_seq_01_v14_multiview_m.npz \
        --teacher outputs/ray_attention_pp_teacher.pth \
        --clip_len 13 --d 64 --epochs 20 \
        --lambda_pseudo 0.5 --pseudo_conf_thresh 5.0 \
        --output outputs/ray_attention_pp_pseudo_student.pth

    # 2. Train the teacher from scratch inside the same run (omit --teacher).
    python experiments/train_pseudo_label_ray_attention_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --unlabeled data/webbridge/mpi_inf_3dhp/s_04_seq_01_v14_multiview_m.npz \
        --teacher_epochs 5 --clip_len 13 --d 64 --epochs 20 \
        --output outputs/ray_attention_pp_pseudo_student.pth
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_cameras_with_delta
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class RandomClipDataset(torch.utils.data.Dataset):
    """Sample random clips from a labeled .npz file (has joints_3d)."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 2000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield non-overlapping clips from a labeled .npz file."""

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


class UnlabeledRandomClipDataset(torch.utils.data.Dataset):
    """Random clips from an unlabeled .npz file (no joints_3d)."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 2000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.K, self.R, self.t


def collate_labeled(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def collate_unlabeled(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    K = torch.stack([b[1] for b in batch], dim=0)
    R = torch.stack([b[2] for b in batch], dim=0)
    t = torch.stack([b[3] for b in batch], dim=0)
    return x, K, R, t


# ---------------------------------------------------------------------------
# Augmentation / utilities
# ---------------------------------------------------------------------------

def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1,
                 outlier_rate: float = 0.02, outlier_scale: float = 100.0):
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(j, n_views, args):
    """Build the student/teacher ray-attention model with PP correction."""
    return RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
        return_pp_delta=args.pp_loss_weight > 0.0 or args.focal_max_scale > 0.0,
    )


@torch.no_grad()
def _per_sample_reprojection_error(pred_3d, points_2d, K, R, t, confidences, eps=1e-6):
    """Per-sample reprojection MSE (B,) for a batch of clips.

    pred_3d:   (B, T, J, 3)
    points_2d: (B, T, V, J, 2)
    K:         (B, V, 3, 3)
    R:         (B, V, 3, 3)
    t:         (B, V, 3)
    confidences: (B, T, V, J)
    """
    B, T, J, _ = pred_3d.shape
    K = K.unsqueeze(1)  # (B, 1, V, 3, 3)
    R = R.unsqueeze(1).unsqueeze(3)  # (B, 1, V, 1, 3, 3)
    t = t.unsqueeze(1).unsqueeze(-2)  # (B, 1, V, 1, 3)
    X = pred_3d.unsqueeze(2).unsqueeze(-1)  # (B, T, 1, J, 3, 1)
    X_cam = (R @ X).squeeze(-1) + t  # (B, T, V, J, 3)
    z = X_cam[..., 2:3]
    proj = (K.unsqueeze(3) @ X_cam.unsqueeze(-1)).squeeze(-1)
    proj_2d = proj[..., :2] / (z.clamp(min=eps))
    diff = proj_2d - points_2d
    sq = (diff ** 2).sum(dim=-1)  # (B, T, V, J)
    weight = confidences
    per_sample = (sq * weight).sum(dim=(-1, -2, -3)) / weight.sum(dim=(-1, -2, -3)).clamp(min=eps)
    return per_sample


@torch.no_grad()
def generate_pseudo_labels(model, loader, device, conf_thresh: float = 5.0):
    """Generate pseudo-labels and confidence weights for unlabeled clips.

    Returns a list of (x, K, R, t, pseudo_y, weight) tuples.  The confidence
    weight for each clip is derived from the teacher's mean reprojection error
    on the original 2-D observations: lower reprojection error -> higher weight.
    """
    model.eval()
    results = []
    for xb, K, R, t in loader:
        xb = xb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        pred, *_ = model(xb, K=K, R=R, t=t)

        # Reprojection error of the teacher on the visible 2-D keypoints.
        points_2d = xb[..., :2]
        conf = xb[..., 2]
        reproj_per_sample = _per_sample_reprojection_error(pred, points_2d, K, R, t, conf)

        # Confidence weight: sigmoid-thresholded inverse reprojection error.
        # ``conf_thresh`` is interpreted in input keypoint units (metres if the
        # dataset is metric, otherwise pixels).  Small error -> weight ~1.
        weight = torch.sigmoid((conf_thresh - reproj_per_sample) / conf_thresh)

        for i in range(xb.size(0)):
            results.append((xb[i].cpu(), K[i].cpu(), R[i].cpu(), t[i].cpu(), pred[i].cpu(), weight[i].cpu()))
    return results


class PseudoLabelDataset(torch.utils.data.Dataset):
    """Wrap pseudo-labels generated by ``generate_pseudo_labels``."""

    def __init__(self, pseudo_items, augment: bool = False, noise_std: float = 0.0, dropout_rate: float = 0.0):
        self.items = pseudo_items
        self.augment = augment
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, K, R, t, y, w = self.items[idx]
        if self.augment:
            if self.noise_std > 0:
                x = x.clone()
                x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * self.noise_std
            if self.dropout_rate > 0:
                x = x.clone()
                mask = (torch.rand(*x.shape[:3], device=x.device) > self.dropout_rate).float()
                x[..., 2] = x[..., 2] * mask
        return x, y, K, R, t, w


def collate_pseudo(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    w = torch.stack([b[5] for b in batch], dim=0)
    return x, y, K, R, t, w


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            xb, yb, K, R, t = batch[:5]
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def train_teacher(model, train_loader, val_loader, args, device):
    """Optional: train the teacher on labeled data for a few epochs."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            K_pert, R_pert, t_pert, _, _ = perturb_cameras_with_delta(
                K, R, t,
                rot_std=0.0,
                trans_std=0.0,
                focal_std=0.0,
                pp_std=args.cam_aug_pp,
            )
            optimizer.zero_grad()
            pred, *_ = model(xb, K=K_pert, R=R_pert, t=t_pert)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
        print(f"Teacher epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    return best_val


def main():
    parser = argparse.ArgumentParser(description="Semi-supervised teacher-student pseudo-label training for MPI-INF-3DHP")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Labeled train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--unlabeled", type=str, nargs="+", required=True, help="Unlabeled .npz files (no joints_3d)")
    parser.add_argument("--teacher", type=str, default=None, help="Path to pre-trained teacher checkpoint")
    parser.add_argument("--teacher_epochs", type=int, default=0, help="If no teacher checkpoint, train a teacher for this many epochs first")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--focal_max_scale", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--pseudo_train_samples", type=int, default=4000)
    parser.add_argument("--lambda_pseudo", type=float, default=0.5, help="Weight for pseudo-label loss")
    parser.add_argument("--pseudo_conf_thresh", type=float, default=5.0, help="Reprojection error threshold (pixels) for pseudo-label confidence")
    parser.add_argument("--pp_loss_weight", type=float, default=0.0)
    parser.add_argument("--cam_aug_pp", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_pp_pseudo_student.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Labeled loaders.
    labeled_train_datasets = [RandomClipDataset(p, args.clip_len, n_samples=args.train_samples) for p in args.train]
    labeled_train_dataset = torch.utils.data.ConcatDataset(labeled_train_datasets)
    labeled_train_loader = torch.utils.data.DataLoader(
        labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_labeled, num_workers=0,
    )

    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.val_stride)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_labeled, num_workers=0,
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}")

    # -------------------------------------------------------------------
    # Teacher model
    # -------------------------------------------------------------------
    teacher = build_model(j, n_views, args).to(device)
    if args.teacher is not None:
        state = torch.load(args.teacher, map_location="cpu", weights_only=True)
        teacher.load_state_dict(state, strict=False)
        print(f"Loaded teacher checkpoint from {args.teacher}")
    elif args.teacher_epochs > 0:
        print(f"Training teacher for {args.teacher_epochs} epochs on labeled data...")
        train_teacher(teacher, labeled_train_loader, val_loader, args, device)
        print("Teacher training complete.")
    else:
        print("WARNING: no teacher checkpoint and --teacher_epochs=0; using random teacher to generate pseudo-labels.")

    # -------------------------------------------------------------------
    # Generate pseudo-labels
    # -------------------------------------------------------------------
    unlabeled_datasets = [UnlabeledRandomClipDataset(p, args.clip_len, n_samples=args.pseudo_train_samples) for p in args.unlabeled]
    unlabeled_dataset = torch.utils.data.ConcatDataset(unlabeled_datasets)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_unlabeled, num_workers=0,
    )
    print("Generating pseudo-labels for unlabeled data...")
    pseudo_items = generate_pseudo_labels(teacher, unlabeled_loader, device, conf_thresh=args.pseudo_conf_thresh)
    print(f"Generated {len(pseudo_items)} pseudo-labeled clips")

    pseudo_dataset = PseudoLabelDataset(pseudo_items, augment=True, noise_std=0.5, dropout_rate=0.05)
    pseudo_loader = torch.utils.data.DataLoader(
        pseudo_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pseudo, num_workers=0,
    )

    # -------------------------------------------------------------------
    # Student model
    # -------------------------------------------------------------------
    student = build_model(j, n_views, args).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # We will iterate labeled and pseudo-labeled loaders in lock-step.
    pseudo_iter = iter(pseudo_loader)
    labeled_iter = iter(labeled_train_loader)

    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0
        n_train_steps = 0

        # Use the labeled loader to define one epoch length.
        for _ in range(len(labeled_train_loader)):
            try:
                x_l, y_l, K_l, R_l, t_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_train_loader)
                x_l, y_l, K_l, R_l, t_l = next(labeled_iter)

            try:
                x_p, y_p, K_p, R_p, t_p, w_p = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                x_p, y_p, K_p, R_p, t_p, w_p = next(pseudo_iter)

            # Move to device.
            x_l, y_l = x_l.to(device), y_l.to(device)
            K_l, R_l, t_l = K_l.to(device), R_l.to(device), t_l.to(device)
            x_p, y_p = x_p.to(device), y_p.to(device)
            K_p, R_p, t_p = K_p.to(device), R_p.to(device), t_p.to(device)
            w_p = w_p.to(device)

            # Augment labeled clip and perturb cameras.
            x_l = augment_clip(x_l)
            x_p = augment_clip(x_p)
            K_l, R_l, t_l, _, _ = perturb_cameras_with_delta(
                K_l, R_l, t_l, rot_std=0.0, trans_std=0.0, focal_std=0.0, pp_std=args.cam_aug_pp
            )
            K_p, R_p, t_p, _, _ = perturb_cameras_with_delta(
                K_p, R_p, t_p, rot_std=0.0, trans_std=0.0, focal_std=0.0, pp_std=args.cam_aug_pp
            )

            optimizer.zero_grad()

            pred_l, *_ = student(x_l, K=K_l, R=R_l, t=t_l)
            loss_l = criterion(pred_l, y_l)

            pred_p, *_ = student(x_p, K=K_p, R=R_p, t=t_p)
            # Weighted MSE: down-weight uncertain pseudo-labels.
            loss_p = (w_p.view(-1, 1, 1, 1) * (pred_p - y_p).pow(2)).mean()

            loss = loss_l + args.lambda_pseudo * loss_p
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * (x_l.size(0) + x_p.size(0))
            n_train_steps += x_l.size(0) + x_p.size(0)

        train_loss /= max(n_train_steps, 1)

        val_err = evaluate(student, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(student.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
