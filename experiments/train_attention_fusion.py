"""Train the lightweight attention fusion model on synthetic multi-view data.

This is a sanity-check experiment: the model learns to fuse 2D + confidence
from a few calibrated views into a 3D skeleton.
"""

from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.attention_model import AttentionFusionModel
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe


def make_cameras(n_views: int = 4, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(123)
    cameras = []
    for i in range(n_views):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[:2, 2] = rng.uniform(300, 340, size=2)
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        radius = 5.0
        c = radius * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def generate_sample(n_views: int = 4, j: int = 17, rng: np.random.Generator = None):
    """Return one synthetic sample: (input, target_3d, cameras)."""
    if rng is None:
        rng = np.random.default_rng(0)
    # Random skeleton centered at origin-ish
    X = rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5])
    X[:, 2] += 3.0  # in front of cameras

    cameras = make_cameras(n_views, rng)
    proj = []
    conf = []
    for cam in cameras:
        P = cam.projection_matrix
        X_h = np.hstack([X, np.ones((j, 1))])
        x_h = (P @ X_h.T).T
        x = x_h[:, :2] / x_h[:, 2:3]
        # Add small image noise
        x += rng.normal(0, 0.5, size=x.shape)
        proj.append(x)
        conf.append(rng.uniform(0.5, 1.0, size=j))
    input_tensor = np.concatenate([np.stack(proj, axis=0), np.stack(conf, axis=0)[..., None]], axis=-1)
    return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(X, dtype=torch.float32), cameras


def generate_dataset(n_samples: int, n_views: int = 4, j: int = 17, seed: int = 0):
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for _ in range(n_samples):
        inp, tgt, _ = generate_sample(n_views, j, rng)
        inputs.append(inp)
        targets.append(tgt)
    return torch.stack(inputs), torch.stack(targets)


def main():
    parser = argparse.ArgumentParser(description="Train attention fusion on synthetic data.")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--j", type=int, default=17)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print("Generating synthetic dataset...")
    X_train, y_train = generate_dataset(200, args.n_views, args.j, seed=0)
    X_val, y_val = generate_dataset(40, args.n_views, args.j, seed=9999)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val), batch_size=args.batch_size
    )

    model = AttentionFusionModel(j=args.j, d=args.d, n_views=args.n_views)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_mpjpe += np.linalg.norm(pred.numpy() - yb.numpy(), axis=-1).mean() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_mpjpe /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "attention_fusion_synthetic.pth")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}")

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / 'attention_fusion_synthetic.pth'}")


if __name__ == "__main__":
    main()
