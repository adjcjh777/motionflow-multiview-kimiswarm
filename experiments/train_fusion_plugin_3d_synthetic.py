"""Train a FusionModule plugin with 3D-supervised synthetic data.

This script trains ``AttentionFusionModule`` (registered under the name
``attention``) using the same synthetic generator as
``train_attention_fusion.py``, but it does so through the ``FusionModule``
plugin interface. After training, ``eval_fusion_plugins_synthetic.py`` can
load the saved checkpoint and compare it with ``DLTFusion``.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_fusion_plugin_3d_synthetic.py
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_attention_fusion import generate_dataset
from motionflow_mv.eval.metrics import mpjpe
from motionflow_mv.fusion import FUSION_REGISTRY
from motionflow_mv.fusion.attention_fusion_module import AttentionFusionModule


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_views, j = 4, 17
    epochs = 50
    lr = 1e-3
    batch_size = 16

    print(f"Device: {device}")

    # Synthetic 3D-supervised dataset.
    X_train, y_train = generate_dataset(200, n_views, j, seed=0)
    X_val, y_val = generate_dataset(40, n_views, j, seed=9999)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size
    )

    # Instantiate the plugin and move underlying model to the training device.
    plugin = AttentionFusionModule(j=j, d=32, n_views=n_views)
    plugin.model.to(device)
    optimizer = torch.optim.Adam(plugin.model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        plugin.model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = plugin.model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        plugin.model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = plugin.model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_mpjpe += mpjpe(pred.cpu().numpy(), yb.cpu().numpy()) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_mpjpe /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(plugin.model.state_dict(), output_dir / "attention_fusion_synthetic_plugin.pth")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}")

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / 'attention_fusion_synthetic_plugin.pth'}")


if __name__ == "__main__":
    main()
