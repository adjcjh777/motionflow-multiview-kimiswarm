"""CPU smoke test for trainer optimization utilities.

Verifies that the prototype utilities in
``motionflow_mv.fusion.prototypes.trainer_optim_utils`` work on CPU:

* linear warmup + cosine LR schedule
* gradient clipping
* AMP context (falls back to no-op on CPU)

Usage
-----
    python experiments/prototypes/smoke_trainer_optim_utils.py

Exit
----
    Exits with code 0 on success, 1 on failure.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.fusion.prototypes.trainer_optim_utils import (
    AMPContext,
    build_lr_scheduler,
    clip_gradients,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def test_warmup_cosine_lr():
    model = _TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_lr_scheduler(optimizer, total_epochs=10, warmup_epochs=3, eta_min=0.0)

    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Warmup: lr should increase from 0.1/3 to 0.1
    assert lrs[0] < lrs[1] < lrs[2], f"Warmup did not increase: {lrs[:3]}"
    assert abs(lrs[2] - 0.1) < 1e-6, f"Warmup did not reach base lr: {lrs[2]}"
    # Post-warmup cosine decay: lr should decrease
    assert lrs[-1] < lrs[3], f"Cosine decay did not decrease: {lrs[3]} vs {lrs[-1]}"
    print("test_warmup_cosine_lr passed")


def test_gradient_clipping():
    model = _TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    x = torch.randn(4, 10)
    y = torch.randn(4, 1)
    optimizer.zero_grad()
    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    loss.backward()

    # Make gradients large by scaling them artificially.
    for p in model.parameters():
        p.grad *= 1e6

    max_norm = 1.0
    grad_norm = clip_gradients(model, max_norm)
    assert grad_norm is not None
    assert grad_norm > max_norm, "Gradients were not large enough to be clipped"

    # After clipping, the total norm should be <= max_norm.
    total_norm = torch.norm(
        torch.stack([p.grad.norm() for p in model.parameters()]),
        p=2,
    )
    assert total_norm.item() <= max_norm + 1e-6, f"Gradient norm after clipping {total_norm.item()} > {max_norm}"
    print("test_gradient_clipping passed")


def test_amp_context_cpu():
    model = _TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    device = torch.device("cpu")
    amp = AMPContext(enabled=True, device=device)

    x = torch.randn(4, 10)
    y = torch.randn(4, 1)

    with amp:
        pred = model(x)
        loss = nn.MSELoss()(pred, y)

    scaled = amp.scale(loss)
    scaled.backward()
    amp.unscale(optimizer)
    clip_gradients(model, 1.0)
    amp.step(optimizer)
    amp.update()

    # On CPU, AMP is disabled (no-op), so parameters should still update.
    for p in model.parameters():
        assert p.grad is not None
    print("test_amp_context_cpu passed")


def test_amp_context_disabled():
    amp = AMPContext(enabled=False, device=torch.device("cpu"))
    x = torch.randn(4, 10)
    y = torch.randn(4, 1)
    model = _TinyModel()
    with amp:
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
    scaled = amp.scale(loss)
    assert scaled is loss
    print("test_amp_context_disabled passed")


def main():
    tests = [
        test_warmup_cosine_lr,
        test_gradient_clipping,
        test_amp_context_cpu,
        test_amp_context_disabled,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            raise
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
