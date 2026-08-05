"""CPU smoke test for the action-aware PP model on a single H36M action.

This script does **not** train to convergence; it only verifies that the
action-aware principal-point model can consume the action-aware H36M dataset,
run a forward/backward step on CPU, and report per-action data distribution.

Example
-------
    python experiments/smoke_action_aware_pp_h36m.py \
        --npz data/h36m_hf/s_01_act_02_multiview.npz \
        --batch_size 2 --n_samples 10 --clip_len 9 --epochs 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.action_aware_dataset import (
    ActionAwareRandomClipDataset,
    action_distribution,
    collate_fn,
)
from motionflow_mv.fusion.action_aware_principal_point_model import (
    ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="data/h36m_hf/s_01_act_02_multiview.npz")
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cpu")
    npz_path = Path(args.npz)

    # Dataset yields (x, y, K, R, t, action_id) tuples.
    dataset = ActionAwareRandomClipDataset(
        str(npz_path),
        clip_len=args.clip_len,
        n_samples=args.n_samples,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    sample = np.load(args.npz)
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    action_id = dataset[0][5]
    print(f"Data: n_views={n_views}, joints={j}, clip_len={args.clip_len}, action_id={action_id}")

    num_actions = max(16, max(action_distribution([args.npz]).keys()))
    print(f"Action vocabulary size (max id): {num_actions}")

    model = ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=2,
        n_joint_layers=1,
        n_st_layers=1,
        max_temporal_len=256,
        residual_hidden=64,
        principal_point_hidden=32,
        principal_point_max_offset=20.0,
        num_actions=num_actions,
        action_embed_dim=args.d,
        return_pp_delta=False,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Action embedding parameters: {sum(p.numel() for p in model.action_embed.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mpjpe = 0.0
        n = 0
        for xb, yb, K, R, t, action in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            pred, _ = model(xb, action_id=action, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_mpjpe += (pred.detach() - yb).norm(dim=-1).mean().item() * xb.size(0)
            n += xb.size(0)

        train_loss /= n
        train_mpjpe /= n
        print(f"Epoch {epoch}: loss={train_loss:.6f}, MPJPE={train_mpjpe:.6f} m")

    print("\n--- Per-file action distribution (frames) ---")
    dist = action_distribution([args.npz], normalize=False)
    for act_id, count in sorted(dist.items()):
        print(f"  action_id={act_id:2d}: {count:6d} frames")

    print("\nCPU smoke test completed successfully.")


if __name__ == "__main__":
    main()
