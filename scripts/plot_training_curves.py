#!/usr/bin/env python
"""Plot training curves from a v5 training log.

Usage:
    python scripts/plot_training_curves.py outputs/omniview_fusion_v26_udp_full_local_4090.log
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path: str):
    text = Path(log_path).read_text(errors="ignore")
    train_steps = re.findall(r"train step (\d+): loss=([\d.]+)", text)
    train_steps = [(int(s), float(l)) for s, l in train_steps]

    val_epochs = re.findall(
        r"Epoch\s+(\d+):\s+train_loss=([\d.]+),\s+val_loss=([\d.]+),\s+val_MPJPE=([\d.]+)mm",
        text,
    )
    val_epochs = [(int(e), float(tl), float(vl), float(mpjpe)) for e, tl, vl, mpjpe in val_epochs]

    return train_steps, val_epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from a v5 log")
    parser.add_argument("log", help="Path to training log")
    parser.add_argument("--out", default="outputs/training_curve.png", help="Output image path")
    args = parser.parse_args()

    train_steps, val_epochs = parse_log(args.log)
    if not train_steps:
        print("No training steps found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if train_steps:
        steps, losses = zip(*train_steps)
        axes[0].plot(steps, losses, label="train loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()

    if val_epochs:
        epochs, train_losses, val_losses, mpjpe = zip(*val_epochs)
        axes[1].plot(epochs, mpjpe, marker="o", label="val MPJPE (mm)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MPJPE (mm)")
        axes[1].set_title("Validation MPJPE")
        axes[1].legend()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved training curve -> {args.out}")


if __name__ == "__main__":
    main()
