"""CPU smoke test for the focal calibration loss.

Summary
-------
Trains a tiny MLP to recover a known focal-length perturbation factor from a
synthetic (perturbed_fx, original_fx) pair.  The test uses the new
``focal_calibration_loss`` and a YAML config, and is intentionally tiny enough
to run on CPU in a few seconds.

Usage
-----
    python experiments/train_focal_calibration_smoke.py --config configs/train_focal_calibration_smoke.yaml

Verification
------------
A successful smoke run prints a decreasing loss and exits with code 0.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.losses import focal_calibration_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


class TinyFocalNet(nn.Module):
    """Tiny MLP that predicts the focal correction scale.

    Input is the (perturbed_fx, original_fx) pair, normalized by a nominal
    focal length.  The model learns to output ``original / perturbed`` so that
    the corrected focal length matches the true one.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # scale > 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, 2)
        return self.mlp(x).squeeze(-1)  # (B, V)


def make_synthetic_data(
    batch_size: int,
    n_views: int,
    focal_max_scale: float,
    seed: int,
):
    """Generate a synthetic batch of perturbed focal lengths and targets.

    Returns:
        features: (B, V, 2) with [perturbed_fx, original_fx] / 1000.
        true_scale: (B, V) perturbation factor p (perturbed = original * p).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Original focal lengths per view, in a realistic range [800, 1200] mm.
    original_fx = torch.rand(batch_size, n_views, generator=generator) * 400.0 + 800.0

    # Random multiplicative perturbation, e.g. +/- focal_max_scale.
    perturb = torch.rand(batch_size, n_views, generator=generator) * 2.0 * focal_max_scale + (
        1.0 - focal_max_scale
    )
    perturbed_fx = original_fx * perturb

    features = torch.stack([perturbed_fx, original_fx], dim=-1) / 1000.0
    return features, perturb


def main():
    parser = argparse.ArgumentParser(description="CPU smoke test for focal calibration loss")
    parser.add_argument("--config", type=str, default="configs/train_focal_calibration_smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    training = cfg.get("training", {})
    focal_cfg = cfg.get("focal", {})
    output_cfg = cfg.get("output", {})

    epochs = training.get("epochs", 20)
    lr = training.get("lr", 0.01)
    batch_size = training.get("batch_size", 64)
    n_views = training.get("n_views", 4)
    seed = training.get("seed", 42)

    focal_max_scale = focal_cfg.get("focal_max_scale", 0.1)
    focal_loss_weight = focal_cfg.get("focal_loss_weight", 1.0)

    output_dir = Path(output_cfg.get("dir", "outputs"))
    checkpoint_name = output_cfg.get("checkpoint_name", "focal_calibration_smoke.pth")

    set_seed(seed)
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Seed:   {seed}")

    model = TinyFocalNet(hidden=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Fixed synthetic batch for reproducibility.
    features, true_scale = make_synthetic_data(
        batch_size=batch_size,
        n_views=n_views,
        focal_max_scale=focal_max_scale,
        seed=seed,
    )
    features = features.to(device)
    true_scale = true_scale.to(device)

    print(f"Synthetic data: features={features.shape}, true_scale={true_scale.shape}")
    print(f"Initial target mean: {true_scale.mean().item():.4f}")

    initial_loss = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_scale = model(features)
        loss = focal_calibration_loss(pred_scale, true_scale)
        loss = focal_loss_weight * loss

        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}: focal_loss={loss.item():.6f}")

    final_loss = loss.item()
    print(f"Initial focal_loss={initial_loss:.6f}, final focal_loss={final_loss:.6f}")

    # Save a tiny checkpoint.
    output_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = output_dir / checkpoint_name
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "final_loss": final_loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")

    # Smoke-test sanity checks.
    assert final_loss < initial_loss, "Loss did not decrease during training"
    assert torch.isfinite(torch.tensor(final_loss)), "Loss is not finite"
    print("focal calibration loss CPU smoke test passed")


if __name__ == "__main__":
    main()
