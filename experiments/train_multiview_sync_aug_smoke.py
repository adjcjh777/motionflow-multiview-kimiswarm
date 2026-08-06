"""CPU smoke run for the multi-view sync-augmentation wrapper.

This script generates a tiny synthetic multi-view sequence, wraps a small
RayAttention model with :class:`MultiViewDataAugmentationWrapper`, and runs a
few training steps.  It is meant to verify that the new augmentation module and
wrapper integrate cleanly with the existing training pipeline.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.synthetic_3d_dataset import generate_sequence
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.models.data_augmentation_multiview_wrapper import (
    MultiViewDataAugmentationWrapper,
)


def build_temp_npz(n_frames: int = 60, n_views: int = 4, j: int = 17):
    inputs, baselines, gt, cameras = generate_sequence(
        n_frames=n_frames, n_views=n_views, j=j, noise_std=1.0
    )
    K = np.stack([cam.K for cam in cameras], axis=0)
    R = np.stack([cam.R for cam in cameras], axis=0)
    t = np.stack([cam.t for cam in cameras], axis=0)
    return {
        "points_2d": inputs[..., :2].numpy(),
        "confidences": inputs[..., 2].numpy(),
        "joints_3d": gt.numpy(),
        "camera_K": K.astype(np.float32),
        "camera_R": R.astype(np.float32),
        "camera_t": t.astype(np.float32),
    }


class RandomClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 200):
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
        start = np.random.randint(0, max(1, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.joints_3d[start:end], self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def main():
    parser = argparse.ArgumentParser(description="Smoke test for multi-view sync augmentation")
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--residual_hidden", type=int, default=32)
    parser.add_argument("--n_st_layers", type=int, default=1)
    parser.add_argument("--clip_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--use_aug", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "smoke.npz"
        data = build_temp_npz(n_frames=60, n_views=4, j=17)
        np.savez(npz_path, **data)

        dataset = RandomClipDataset(str(npz_path), args.clip_len, n_samples=args.train_samples)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )

        base_model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
            j=17,
            d=args.d,
            n_views=4,
            n_st_layers=args.n_st_layers,
            n_heads=2,
            n_joint_layers=1,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=16,
            principal_point_max_offset=10.0,
            return_pp_delta=False,
        ).to(device)

        if args.use_aug:
            model = MultiViewDataAugmentationWrapper(base_model)
        else:
            model = base_model

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        print(f"Device: {device}, use_aug={args.use_aug}, params={sum(p.numel() for p in model.parameters())}")
        model.train()
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            for xb, yb, K, R, t in loader:
                xb, yb = xb.to(device), yb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                optimizer.zero_grad()
                pred, _ = model(xb, K=K, R=R, t=t)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch}: loss={epoch_loss:.6f}")

        model.eval()
        with torch.no_grad():
            xb, yb, K, R, t = next(iter(loader))
            pred, _ = model(xb.to(device), K=K.to(device), R=R.to(device), t=t.to(device))
            print(f"Eval forward shape: {pred.shape}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
