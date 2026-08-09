"""Run a single training step and check for NaN model weights."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
torch.autograd.set_detect_anomaly(True)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from train_omniview_fusion_v5_webbridge_multi import (
    OmniMultiViewTrainer,
    build_compute_loss,
    build_datasets,
    build_model_from_args,
    webbridge_mixed_collate_fn,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    train_args = argparse.Namespace(**cfg)
    if os.name == "nt":
        train_args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dataset, n_views, n_joints = build_datasets(train_args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        collate_fn=webbridge_mixed_collate_fn,
        num_workers=0,
    )
    model = build_model_from_args(train_args, n_joints, n_views, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)
    trainer = OmniMultiViewTrainer(
        model,
        optimizer,
        device,
        train_args,
        total_epochs=1,
        max_grad_norm=train_args.max_grad_norm,
        ema_decay=train_args.ema_decay,
    )

    def count_nan_params(model):
        nan_count = 0
        for p in model.parameters():
            if p.requires_grad and torch.isnan(p).any():
                nan_count += 1
        return nan_count

    for step in range(args.steps):
        batch = next(iter(val_loader))
        loss, metrics = trainer.compute_loss(model, batch, device)
        print(f"Step {step}: loss={loss.item():.6f} nan_loss={torch.isnan(loss).item()}")
        loss.backward()
        nan_grad_names = [
            name for name, p in model.named_parameters()
            if p.grad is not None and torch.isnan(p.grad).any()
        ]
        print(f"  NaN grads: {len(nan_grad_names)}; first few: {nan_grad_names[:5]}")
        optimizer.step()
        print(f"  NaN params after step: {count_nan_params(model)}")


if __name__ == "__main__":
    main()
