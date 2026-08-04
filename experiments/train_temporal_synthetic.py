"""Pre-train TemporalRefinerModel on synthetic temporal sequences.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_temporal_synthetic.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.temporal_refiner import TemporalRefinerModel


def make_cameras(n_views: int = 5, rng: np.random.Generator = None):
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


def generate_sequence(n_frames: int = 10, n_views: int = 5, j: int = 17, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(0)
    # Generate a smooth base skeleton trajectory.
    base = rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5])
    base[:, 2] += 3.0
    positions = [base.copy()]
    for _ in range(n_frames - 1):
        delta = rng.normal(0, 0.05, size=base.shape)
        base = base + delta
        positions.append(base.copy())
    positions = np.stack(positions, axis=0)  # (T, J, 3)

    cameras = make_cameras(n_views, rng)
    proj = [cam.projection_matrix for cam in cameras]

    inputs = []
    baselines = []
    for t in range(n_frames):
        X = positions[t]
        points_2d = []
        conf = []
        for cam in cameras:
            P = cam.projection_matrix
            X_h = np.hstack([X, np.ones((j, 1))])
            x_h = (P @ X_h.T).T
            x = x_h[:, :2] / x_h[:, 2:3]
            x += rng.normal(0, 1.0, size=x.shape)
            points_2d.append(x)
            conf.append(rng.uniform(0.5, 1.0, size=j))
        points_2d = np.stack(points_2d, axis=0)  # (V, J, 2)
        conf = np.stack(conf, axis=0)  # (V, J)
        inputs.append(np.concatenate([points_2d, conf[..., None]], axis=-1))

        # Baseline DLT from noisy points
        baseline = np.zeros((j, 3), dtype=np.float64)
        for joint_idx in range(j):
            A = []
            for v, cam in enumerate(cameras):
                P = cam.projection_matrix
                u, v_coord = points_2d[v, joint_idx]
                A.append(u * P[2] - P[0])
                A.append(v_coord * P[2] - P[1])
            A = np.stack(A)
            _, _, vt = np.linalg.svd(A)
            X_h = vt[-1]
            baseline[joint_idx] = X_h[:3] / X_h[3]
        baselines.append(baseline)

    return (torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32),
            torch.tensor(np.stack(baselines, axis=0), dtype=torch.float32),
            torch.tensor(positions, dtype=torch.float32))


def generate_dataset(n_seq: int, n_frames: int = 10, n_views: int = 5, j: int = 17, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, B, Y = [], [], []
    for _ in range(n_seq):
        inp, base, gt = generate_sequence(n_frames, n_views, j, rng)
        X.append(inp)
        B.append(base)
        Y.append(gt)
    return torch.stack(X), torch.stack(B), torch.stack(Y)


def main():
    parser = argparse.ArgumentParser(description="Pre-train temporal refiner on synthetic data.")
    parser.add_argument("--n_seq", type=int, default=200)
    parser.add_argument("--n_frames", type=int, default=9)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="outputs/temporal_refiner_synthetic.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Generating synthetic dataset...")
    X, B, Y = generate_dataset(args.n_seq, args.n_frames, 5, 17, seed=0)
    print(f"Dataset shape: inputs {X.shape}, baselines {B.shape}, targets {Y.shape}")

    train_dataset = torch.utils.data.TensorDataset(X, B, Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = TemporalRefinerModel(j=17, d=args.d, n_views=5, hidden=args.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, bb, yb in train_loader:
            xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, bb)
            center = xb.shape[1] // 2
            loss = criterion(pred, yb[:, center])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_loader.dataset)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), output_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: loss={total_loss:.4f}")

    print(f"Best loss={best_loss:.4f}, checkpoint: {output_path}")


if __name__ == "__main__":
    main()
