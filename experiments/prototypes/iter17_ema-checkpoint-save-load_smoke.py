"""Smoke test for EMA checkpoint save/load on a tiny multi-view pose model.

This prototype trains ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
on the same synthetic multi-view dataset used by the Bayesian triangulation v3 smoke
script.  It maintains an exponential moving average (EMA) of the model weights,
saves a checkpoint containing both the online and EMA weights, and then reloads the
checkpoint into a fresh model to verify that the EMA copy is restored correctly.

Usage
-----
    python experiments/prototypes/iter17_ema-checkpoint-save-load_smoke.py
    python experiments/prototypes/iter17_ema-checkpoint-save-load_smoke.py --epochs 2 --batch_size 2

Output
------
    - prints per-epoch train/val metrics (online and EMA)
    - saves a checkpoint to ``outputs/iter17_ema_checkpoint_smoke.pth``
    - reloads the checkpoint and verifies forward consistency
"""

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_synthetic_cameras(n_views: int = 4):
    """Build a circular rig of pinhole cameras and return K, R, t tensors."""
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    K = torch.from_numpy(np.stack(K_list, axis=0)).float()
    R = torch.from_numpy(np.stack(R_list, axis=0)).float()
    t = torch.from_numpy(np.stack(t_list, axis=0)).float()
    return K, R, t


def _project_points(joints_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project world points into all views."""
    X = joints_3d  # (F, J, 3)
    t = t[:, None, None, :]  # (V, 1, 1, 3)
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t  # (V, F, J, 3)
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])  # (V, F, J, 3, 1)
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)  # (F, V, J, 2)


class EMA:
    """Exponential moving average of model parameters.

    Parameters
    ----------
    model:
        Model whose trainable parameters will be shadowed.
    decay:
        EMA decay coefficient (close to 1.0 means slow averaging).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.setup(model)

    def setup(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Call after each optimizer step to update the EMA shadow copy."""
        one_minus_decay = 1.0 - self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                    continue
                new_average = self.decay * self.shadow[name] + one_minus_decay * param.data
                self.shadow[name] = new_average

    def copy_to(self, model: nn.Module):
        """Copy the EMA shadow parameters into the model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def store(self, model: nn.Module):
        """Snapshot the current online parameters (e.g. before copying EMA in)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def restore(self, model: nn.Module):
        """Restore the online parameters saved by ``store``."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.shadow = {k: v.clone() for k, v in state_dict["shadow"].items()}


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset for smoke tests."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 100,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3  # (F, J, 3)
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d  # (F, V, J, 2)
        self.confidences = torch.ones_like(points_2d[..., 0])  # (F, V, J)
        self.joints_3d = joints_3d  # (F, J, 3)

        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - clip_len + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx
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


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    total_err = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            count += xb.size(0)
    return total_err / count


def save_checkpoint(path: Path, model: nn.Module, ema: EMA, epoch: int, best_val: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, ema: EMA):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    return state


def main():
    parser = argparse.ArgumentParser(description="Smoke test for EMA checkpoint save/load")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--d", type=int, default=32, help="Feature dimension")
    parser.add_argument("--n_st_layers", type=int, default=1, help="Spatio-temporal transformer layers")
    parser.add_argument("--residual_hidden", type=int, default=32, help="Residual MLP hidden size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--ema_decay", type=float, default=0.9, help="EMA decay coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--output", type=str, default="outputs/iter17_ema_checkpoint_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17
    n_train_frames = 120
    n_val_frames = 40

    train_dataset = SyntheticSmokeDataset(K, R, t, n_frames=n_train_frames, n_joints=n_joints, clip_len=args.clip_len)
    val_dataset = SyntheticSmokeDataset(K, R, t, n_frames=n_val_frames, n_joints=n_joints, clip_len=args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=32,
        principal_point_max_offset=5.0,
        return_pp_delta=False,
    ).to(device)

    ema = EMA(model, decay=args.ema_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"n_views=4, j={n_joints}, clip_len={args.clip_len}, d={args.d}, "
        f"n_st_layers={args.n_st_layers}, residual_hidden={args.residual_hidden}, "
        f"params={n_params}, ema_decay={args.ema_decay}"
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, Kb, Rb, tb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)

            optimizer.zero_grad()
            pred, *_ = model(xb, K=Kb, R=Rb, t=tb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            ema.update(model)
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Evaluate online, then EMA, then restore online weights.
        online_val_err = evaluate(model, val_loader, device)

        ema.store(model)
        ema.copy_to(model)
        ema_val_err = evaluate(model, val_loader, device)
        ema.restore(model)

        # Save the *online* weights; EMA state is also persisted.
        if ema_val_err < best_val:
            best_val = ema_val_err
            save_checkpoint(output_path, model, ema, epoch, best_val)
            saved_marker = " (saved)"
        else:
            saved_marker = ""

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_MPJPE_online={online_val_err*1000:.2f}mm, "
            f"val_MPJPE_ema={ema_val_err*1000:.2f}mm{saved_marker}"
        )

    # Verify checkpoint save/load by reloading into a fresh model + EMA.
    print(f"\nReloading checkpoint from {output_path} ...")
    fresh_model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=32,
        principal_point_max_offset=5.0,
        return_pp_delta=False,
    ).to(device)
    fresh_ema = EMA(fresh_model, decay=args.ema_decay)
    load_checkpoint(output_path, fresh_model, fresh_ema)
    online_state = copy.deepcopy(fresh_model.state_dict())

    loaded_val_err = evaluate(fresh_model, val_loader, device)
    print(f"Reloaded online model val_MPJPE={loaded_val_err*1000:.2f}mm")

    fresh_ema.copy_to(fresh_model)
    ema_loaded_val_err = evaluate(fresh_model, val_loader, device)
    print(f"Reloaded EMA model val_MPJPE={ema_loaded_val_err*1000:.2f}mm")

    # Restore online weights for the sanity check.
    fresh_model.load_state_dict(online_state)

    # Sanity check: loaded online weights produce the same predictions.
    with torch.no_grad():
        sample_x, sample_y, sample_K, sample_R, sample_t = next(iter(val_loader))
        sample_x = sample_x.to(device)
        sample_K, sample_R, sample_t = sample_K.to(device), sample_R.to(device), sample_t.to(device)
        pred_original, *_ = model(sample_x, K=sample_K, R=sample_R, t=sample_t)
        pred_loaded, *_ = fresh_model(sample_x, K=sample_K, R=sample_R, t=sample_t)
        max_diff = (pred_original - pred_loaded).abs().max().item()
        print(f"Max online prediction difference after reload: {max_diff:.2e}")
        assert max_diff < 1e-5, "Online predictions diverged after checkpoint reload!"

    print(f"\nBest val MPJPE (EMA): {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
