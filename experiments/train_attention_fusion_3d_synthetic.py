"""Train AttentionFusionModel as a FusionModule plugin on 3D synthetic data.

Uses the shared synthetic_3d_dataset generator so the checkpoint is compatible
with eval_all_plugins_synthetic.py.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_attention_fusion_3d_synthetic.py
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.synthetic_3d_dataset import generate_dataset
from motionflow_mv.fusion.attention_model import AttentionFusionModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_seq, n_frames, n_views, j = 200, 9, 5, 17
    epochs = 50
    lr = 1e-3
    batch_size = 16

    print(f"Device: {device}")
    print("Generating synthetic dataset...")
    X, B, Y = generate_dataset(n_seq, n_frames, n_views, j, seed=0, noise_std=1.0)
    # Flatten sequence dimension: each frame is a separate sample.
    X = X.reshape(-1, n_views, j, 3)
    Y = Y.reshape(-1, j, 3)
    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AttentionFusionModel(j=j, d=32, n_views=n_views).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_loader.dataset)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), output_dir / "attention_fusion_synthetic_shared.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: loss={total_loss:.4f}")

    print(f"Best loss={best_loss:.4f}, checkpoint: {output_dir / 'attention_fusion_synthetic_shared.pth'}")


if __name__ == "__main__":
    main()
