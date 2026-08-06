"""Train a domain-adaptive temporal-residual model for synthetic-to-real transfer.

This script loads synthetic (source) and real (target) canonical .npz clips and
fine-tunes ``DomainAdaptationWrapper``.  The wrapper adds a GRL-based domain
discriminator and optional domain-specific FiLM adapters on top of the current
best ``RayAttentionFusionModelTemporalResidual`` backbone without modifying it.

Usage (CPU/small-data smoke example)
------------------------------------
    python experiments/train_domain_adapt_mpiinf3dhp.py \
        --synthetic_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
        --real_train data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
        --clip_len 13 --epochs 2 --batch_size 2 --train_samples 100

The real data can be unlabeled by passing ``--unlabeled_real``.  In that case
only the domain discriminator loss is applied to the real clips; the pose loss
is computed only on synthetic clips (and labeled real clips otherwise).
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

from motionflow_mv.fusion.domain_adaptation_wrapper import (
    DomainAdaptationWrapper,
    maximum_mean_discrepancy,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandomClipDataset(torch.utils.data.Dataset):
    """Sample random clips from a canonical .npz sequence.

    Returns ``(x, y, K, R, t)`` where ``x`` is (T, V, J, 3) and ``y`` is
    (T, J, 3).
    """

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
            [
                self.points_2d[start:end],
                self.confidences[start:end].unsqueeze(-1),
            ],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield strided clips from a canonical .npz sequence."""

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


def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1,
                 outlier_rate: float = 0.02, outlier_scale: float = 100.0):
    """Lightweight per-clip augmentation."""
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
            > dropout_rate
        ).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
            < outlier_rate
        )
        outlier = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, device=x.device)
            - 0.5
        ) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


class MixedDomainDataset(torch.utils.data.Dataset):
    """Mix source (synthetic) and target (real) clips with domain labels.

    Returns ``(x, y, K, R, t, domain_label)`` where ``domain_label`` is 0 for
    synthetic and 1 for real.
    """

    def __init__(self, source_paths, target_paths, clip_len: int, n_samples: int = 2000):
        self.source = [RandomClipDataset(p, clip_len, n_samples=n_samples) for p in source_paths]
        self.target = [RandomClipDataset(p, clip_len, n_samples=n_samples) for p in target_paths]
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Sample roughly balanced batches by alternating domain.
        if random.random() < 0.5:
            ds = random.choice(self.source)
            domain = 0
        else:
            ds = random.choice(self.target)
            domain = 1
        x, y, K, R, t = ds[random.randint(0, len(ds) - 1)]
        return x, y, K, R, t, torch.tensor(domain, dtype=torch.long)


def collate_mixed(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    domain_labels = torch.stack([b[5] for b in batch], dim=0)
    return x, y, K, R, t, domain_labels


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            xb, yb, K, R, t = batch[:5]
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(description="Train domain-adaptive synthetic-to-real model on MPI-INF-3DHP")
    parser.add_argument("--synthetic_train", type=str, nargs="+", required=True, help="Synthetic/source .npz files")
    parser.add_argument("--real_train", type=str, nargs="+", required=True, help="Real/target .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=2000, help="Random clips per train file")
    parser.add_argument("--lambda_domain", type=float, default=0.1, help="Weight for domain adversarial loss")
    parser.add_argument("--lambda_mmd", type=float, default=0.0, help="Weight for MMD feature-alignment loss")
    parser.add_argument("--unlabeled_real", action="store_true", help="Treat real clips as unlabeled")
    parser.add_argument("--no_film", action="store_true", help="Disable domain-specific FiLM adapters")
    parser.add_argument("--no_domain_classifier", action="store_true", help="Disable domain discriminator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_residual_domain_adapt.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = MixedDomainDataset(
        args.synthetic_train,
        args.real_train,
        args.clip_len,
        n_samples=args.train_samples,
    )
    val_dataset = TemporalClipDataset(args.val, args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mixed,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Infer dimensions from the first synthetic file.
    sample = np.load(args.synthetic_train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}")

    model = DomainAdaptationWrapper(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
        use_domain_classifier=not args.no_domain_classifier,
        use_domain_film=not args.no_film,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t, domain_labels in train_loader:
            xb = augment_clip(xb)
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            domain_labels = domain_labels.to(device)

            # Mask for labeled samples.  Real samples are labeled only if
            # --unlabeled_real is not set.
            labeled_mask = torch.ones(xb.size(0), device=device, dtype=torch.bool)
            if args.unlabeled_real:
                labeled_mask = domain_labels == 0

            optimizer.zero_grad()
            pred, _, dlogits = model(
                xb, K=K, R=R, t=t, domain_labels=domain_labels, return_domain_logits=True
            )

            # Pose regression loss on labeled clips only.
            pose_loss = torch.tensor(0.0, device=device)
            if labeled_mask.any():
                pose_loss = criterion(pred[labeled_mask], yb[labeled_mask])

            # Domain adversarial loss on all clips.
            domain_loss = torch.tensor(0.0, device=device)
            if dlogits is not None and not args.no_domain_classifier:
                # Domain labels are expanded to per-frame tokens by the wrapper,
                # so we compare to the per-token labels here.
                B, T = xb.size(0), xb.size(1)
                target = domain_labels.unsqueeze(1).expand(B, T).reshape(-1).to(device)
                domain_loss = F.cross_entropy(dlogits, target)

            loss = pose_loss + args.lambda_domain * domain_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err * 1000:.2f}mm (saved)"
            )
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err * 1000:.2f}mm")

    print(f"Best val MPJPE: {best_val * 1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
