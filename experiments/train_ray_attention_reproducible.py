"""Config-driven, reproducible trainer for RayAttentionFusionModelV3.

Summary
-------
This script adds a reproducibility harness around the existing v3 training loop:
* YAML config via `--config` (see `configs/deprecated/circular/train_ray_attention_reproducible.yaml`)
* Deterministic seeding (`torch`, `numpy`, `random`, `cudnn`)
* Optional Weights & Biases logging when `wandb.enabled: true`
* H36M NPZ input with the same `CameraDataset` / `augment_batch` used by
  `train_ray_attention_v3_h36m.py`

Usage
-----
    python experiments/train_ray_attention_reproducible.py \
        --config configs/deprecated/circular/train_ray_attention_reproducible.yaml

Verification
------------
A small smoke-run was performed with `epochs=2` on the local WSL 4090 using
`data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz`.
The script parsed the YAML, seeded RNGs, and completed both training epochs
(Device: cuda, n_views=4, j=17, Model params=93537, epoch-1 val_MPJPE10.0 m).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# YAML is part of the standard library only from Python 3.11 onward; keep the
# import explicit so missing PyYAML fails loudly with a helpful message.
try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    """Make torch, numpy, and random deterministic for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def get_with_defaults(cfg: dict):
    """Return a flat namespace of training hyperparameters with sane defaults."""
    training = cfg.get("training", {})
    aug = cfg.get("augmentation", {})
    wandb = cfg.get("wandb", {})
    output = cfg.get("output", {})

    return argparse.Namespace(
        dataset=cfg.get("dataset", "data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz"),
        d=training.get("d", 64),
        epochs=training.get("epochs", 50),
        lr=training.get("lr", 1e-3),
        batch_size=training.get("batch_size", 32),
        val_ratio=training.get("val_ratio", 0.1),
        seed=training.get("seed", 42),
        noise_std=aug.get("noise_std", 0.5),
        dropout_rate=aug.get("dropout_rate", 0.1),
        outlier_rate=aug.get("outlier_rate", 0.02),
        outlier_scale=aug.get("outlier_scale", 100.0),
        wandb_enabled=wandb.get("enabled", False),
        wandb_project=wandb.get("project", "motionflow-multiview"),
        wandb_entity=wandb.get("entity", None),
        wandb_run_name=wandb.get("run_name", None),
        output_dir=output.get("dir", "outputs"),
        checkpoint_name=output.get("checkpoint_name", "ray_attention_v3_reproducible.pth"),
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Reproducible training for RayAttentionFusionModelV3")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hparams = get_with_defaults(cfg)

    set_seed(hparams.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed:   {hparams.seed}")

    # Optional W&B logging
    wandb = None
    if hparams.wandb_enabled:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError("wandb is enabled in config but not installed: pip install wandb") from exc

        wandb_module.init(
            project=hparams.wandb_project,
            entity=hparams.wandb_entity,
            name=hparams.wandb_run_name,
            config={k: v for k, v in vars(hparams).items()},
        )
        wandb = wandb_module

    data = np.load(hparams.dataset)
    n = data["joints_3d"].shape[0]
    n_val = int(n * hparams.val_ratio)
    perm = np.random.permutation(n)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_dataset = CameraDataset(data, train_idx)
    val_dataset = CameraDataset(data, val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        collate_fn=collate_fn,
    )

    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, K_shape={data['camera_K'].shape}")
    model = RayAttentionFusionModelV3(j=j, d=hparams.d, n_views=n_views).to(device)
    print(f"fusion_mlp weight shape: {model.fusion_mlp[0].weight.shape}")
    optimizer = optim.Adam(model.parameters(), lr=hparams.lr)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    output_dir = Path(hparams.output_dir)
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, hparams.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_batch(
                xb,
                noise_std=hparams.noise_std,
                dropout_rate=hparams.dropout_rate,
                outlier_rate=hparams.outlier_rate,
                outlier_scale=hparams.outlier_scale,
            )
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
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
            ckpt_path = output_dir / hparams.checkpoint_name
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}m")

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mpjpe_m": val_mpjpe,
                    "best/val_loss": best_val,
                },
                step=epoch,
            )

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / hparams.checkpoint_name}")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
