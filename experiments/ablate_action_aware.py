"""CPU smoke test for the action-aware H36M dataset.

Loads a tiny Human3.6M .npz, builds action-aware train/val clips, trains a
dummy model that injects an action embedding, and reports the action
 distribution across the supplied files.

Example
-------
    python experiments/ablate_action_aware.py \
        --train data/webbridge/h36m/s_01_acts_02_multiview.npz \
                data/webbridge/h36m/s_01_acts_03_multiview.npz \
        --val data/webbridge/h36m/s_01_acts_04_multiview.npz \
        --epochs 2 --batch_size 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.action_aware_dataset import (
    ACTION_NAMES,
    ActionAwareRandomClipDataset,
    ActionAwareTemporalClipDataset,
    action_distribution,
)


class TinyActionModel(nn.Module):
    """Minimal model that concatenates an action embedding to the input."""

    def __init__(self, num_actions: int, embed_dim: int, j: int, out_dim: int = 3):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions + 1, embed_dim)
        # Per-joint projection from (2D/conf features + embedding) to 3D joint position.
        self.fc = nn.Linear(3 + embed_dim, out_dim)
        self.j = j
        self.out_dim = out_dim

    def forward(self, x, action_id):
        # x: (B, T, V, J, 3), action_id: (B,)
        B, T, V, J, _ = x.shape
        emb = self.action_embed(action_id)  # (B, embed_dim)
        emb = emb[:, None, None, None, :].expand(B, T, V, J, -1)
        x_feat = torch.cat([x, emb], dim=-1)  # (B, T, V, J, 3 + embed_dim)
        # Aggregate across views; keep per-joint structure.
        x_pooled = x_feat.mean(dim=2)  # (B, T, J, 3 + embed_dim)
        out = self.fc(x_pooled)  # (B, T, J, out_dim)
        return out


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    action = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, nargs="+", required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--embed_dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cpu")

    train_datasets = [ActionAwareRandomClipDataset(p, args.clip_len, n_samples=50) for p in args.train]
    val_dataset = ActionAwareTemporalClipDataset(args.val, args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_datasets),
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
    num_actions = max(16, max(action_distribution(args.train + [args.val]).keys()))

    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, embed_dim={args.embed_dim}")
    print(f"action set size (max action id): {num_actions}")

    model = TinyActionModel(num_actions=num_actions, embed_dim=args.embed_dim, j=j).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"TinyActionModel params: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb, _, _, _, action in train_loader:
            xb, yb, action = xb.to(device), yb.to(device), action.to(device)
            optimizer.zero_grad()
            pred = model(xb, action)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _, _, _, action in val_loader:
                xb, yb, action = xb.to(device), yb.to(device), action.to(device)
                pred = model(xb, action)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}: train_loss={total_loss:.6f}, val_loss={val_loss:.6f}")

    print("\n--- Action distribution (frames) ---")
    dist = action_distribution(args.train + [args.val], normalize=False)
    norm = action_distribution(args.train + [args.val], normalize=True)
    for act_id in sorted(dist.keys()):
        name = ACTION_NAMES.get(act_id, "Unknown")
        print(f"  action {act_id:2d} ({name:12s}): {dist[act_id]:6d} frames ({norm[act_id]:.3f})")

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()
