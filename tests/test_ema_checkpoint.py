"""CPU smoke test for EMA checkpoint save/load.

Verifies that the EMA shadow is updated, saved, loaded, and can be applied to
and restored from a model.  Keeps the model tiny so the test runs in <1 s on
CPU.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.prototypes.ema import (
    EMA,
    load_checkpoint_with_ema,
    save_checkpoint_with_ema,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3, bias=True)

    def forward(self, x):
        return self.fc(x)


def test_ema_update_and_shadow():
    model = TinyModel()
    # Fix seed for reproducibility.
    torch.manual_seed(0)
    ema = EMA(model, decay=0.9, start_epoch=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # Capture initial parameter values.
    initial_weight = model.fc.weight.data.clone()

    for _ in range(5):
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()
        ema.update(model)

    # After training, online parameters should differ from the initial values.
    assert not torch.allclose(model.fc.weight.data, initial_weight, atol=1e-6)

    # EMA shadow should be a convex combination of online values; applying it
    # changes the model weights.
    online_weight = model.fc.weight.data.clone()
    ema.apply_shadow(model)
    shadow_weight = model.fc.weight.data.clone()
    assert not torch.allclose(shadow_weight, online_weight, atol=1e-6)
    ema.restore(model)
    assert torch.allclose(model.fc.weight.data, online_weight, atol=1e-6)


def test_ema_checkpoint_save_load():
    model = TinyModel()
    torch.manual_seed(1)
    ema = EMA(model, decay=0.9, start_epoch=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    for step in range(5):
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()
        # start_epoch=1, so updates happen for all steps (epoch=step+1 >= 1).
        ema.update(model, epoch=step + 1)

    # Save checkpoint with shadow applied, so the saved model key holds EMA weights.
    ema.apply_shadow(model)
    try:
        checkpoint_path = Path("outputs/test_ema_checkpoint.pth")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint_with_ema(
            checkpoint_path, model, ema=ema, optimizer=optimizer,
            epoch=5, best_val=0.123,
        )
    finally:
        ema.restore(model)

    # Load into a fresh model+EMA.
    model2 = TinyModel()
    ema2 = EMA(model2, decay=0.9, start_epoch=1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    loaded_epoch, loaded_best = load_checkpoint_with_ema(
        checkpoint_path, model2, ema=ema2, optimizer=optimizer2, strict=True,
    )

    assert loaded_epoch == 5
    assert abs(loaded_best - 0.123) < 1e-6

    # The loaded model should match the original EMA shadow.
    ema2.apply_shadow(model2)
    for name, p2 in model2.named_parameters():
        p1 = ema.shadow[name]
        assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch for {name}"

    # Clean up.
    checkpoint_path.unlink(missing_ok=True)


def test_ema_start_epoch_skips_early_updates():
    model = TinyModel()
    ema = EMA(model, decay=0.9, start_epoch=3)
    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    # Simulate epoch 1 and 2; shadow should remain at initial values.
    for epoch in [1, 2]:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model, epoch=epoch)

    for k in ema.shadow:
        assert torch.allclose(ema.shadow[k], initial_shadow[k], atol=1e-6)

    # Epoch 3 should update the shadow.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model, epoch=3)
    updated = False
    for k in ema.shadow:
        if not torch.allclose(ema.shadow[k], initial_shadow[k], atol=1e-6):
            updated = True
    assert updated


def test_load_plain_state_dict_still_works():
    model = TinyModel()
    torch.manual_seed(2)
    for p in model.parameters():
        p.data.normal_()
    plain_path = Path("outputs/test_plain_state_dict.pth")
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), plain_path)

    model2 = TinyModel()
    epoch, best = load_checkpoint_with_ema(plain_path, model2)
    assert epoch is None and best is None
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)
    plain_path.unlink(missing_ok=True)


def main():
    test_ema_update_and_shadow()
    test_ema_checkpoint_save_load()
    test_ema_start_epoch_skips_early_updates()
    test_load_plain_state_dict_still_works()
    print("EMA checkpoint save/load smoke tests passed.")


if __name__ == "__main__":
    main()
