"""CPU smoke tests for motionflow_mv.training.trainer_v2.

Covers TrainerV2, MultiViewPoseTrainerV2, EMA state round-trip, AMP on CPU,
and the warmup + cosine LR scheduler.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.training import EMA, TrainerV2, MultiViewPoseTrainerV2, build_lr_scheduler


class DummyModel(nn.Module):
    """Tiny model that maps input to output via a learnable linear layer."""

    def __init__(self, in_features: int = 3, out_features: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class DummyMultiViewModel(nn.Module):
    """Tiny model matching the (x, K, R, t) -> (pred_3d, weights) convention."""

    def __init__(self, j=17):
        super().__init__()
        self.fc = nn.Linear(3, 3)
        self.j = j

    def forward(self, x, K=None, R=None, t=None):
        # x: (B, T, V, J, 3) -> use confidence channel to predict 3D joints
        b, t, v, j, _ = x.shape
        feat = x[..., 2:3]  # (B, T, V, J, 1)
        feat = feat.mean(dim=2)  # (B, T, J, 1)
        pred = self.fc(feat.expand(b, t, j, 3))
        return pred, torch.ones(b, t, v, j)


def _make_dataloader(batch_size=2, num_batches=3):
    data = [(torch.randn(batch_size, 4, 3), torch.randn(batch_size, 4, 3)) for _ in range(num_batches)]
    return torch.utils.data.DataLoader(data, batch_size=1)


def test_lr_scheduler():
    model = nn.Linear(2, 1)
    opt = optim.SGD(model.parameters(), lr=1.0)
    sched = build_lr_scheduler(opt, total_epochs=10, warmup_epochs=2, eta_min=0.0)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    # The scheduler constructor performs one step, so lrs[0] is already
    # inside the warmup, lrs[1] is the peak, and the LR decays afterwards.
    assert lrs[0] < lrs[1]
    assert abs(lrs[1] - 1.0) < 1e-6
    # Last epoch should be close to eta_min
    assert lrs[-1] < 0.1


def test_ema_state():
    model = DummyModel()
    ema = EMA(model, decay=0.9)
    opt = optim.SGD(model.parameters(), lr=0.1)
    for _ in range(5):
        loss = model(torch.randn(2, 3)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
    sd = ema.state_dict()
    ema2 = EMA(model, decay=0.5)
    ema2.load_state_dict(sd)
    for k in ema.shadow:
        assert torch.allclose(ema.shadow[k], ema2.shadow[k])


def test_trainer_v2_basic():
    device = torch.device("cpu")
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    def compute_loss(model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
        return nn.functional.mse_loss(pred, y), {}

    trainer = TrainerV2(
        model,
        optimizer,
        device,
        compute_loss=compute_loss,
        total_epochs=3,
        warmup_epochs=1,
        max_grad_norm=1.0,
        amp_enabled=True,
        ema_decay=0.9,
    )
    loader = _make_dataloader()
    history = trainer.fit(loader, epochs=2, save_best=False)
    assert len(history) == 2
    assert "loss" in history[0]["train"]


def test_multiview_trainer():
    device = torch.device("cpu")
    model = DummyMultiViewModel(j=17)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    trainer = MultiViewPoseTrainerV2(
        model,
        optimizer,
        device,
        total_epochs=2,
        warmup_epochs=0,
        max_grad_norm=5.0,
        amp_enabled=True,
        ema_decay=0.99,
        ema_eval=True,
    )
    # Synthetic (x, y, K, R, t) batch
    x = torch.randn(2, 3, 4, 17, 3)
    y = torch.randn(2, 3, 17, 3)
    K = torch.eye(3).unsqueeze(0).expand(4, -1, -1).float()
    R = torch.eye(3).unsqueeze(0).expand(4, -1, -1).float()
    t = torch.zeros(4, 3).float()
    batches = [(x, y, K, R, t)]
    history = trainer.fit(batches, val_loader=batches, epochs=1, save_best=False)
    assert len(history) == 1
    assert "train" in history[0]
    assert "val" in history[0]


def test_checkpoint_roundtrip(tmp_path=None):
    device = torch.device("cpu")
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    def compute_loss(model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
        return nn.functional.mse_loss(pred, y), {}

    trainer = TrainerV2(
        model,
        optimizer,
        device,
        compute_loss=compute_loss,
        total_epochs=2,
        ema_decay=0.9,
    )
    loader = _make_dataloader()
    trainer.fit(loader, epochs=1)

    path = tmp_path / "ckpt.pth" if tmp_path else "outputs/test_trainer_v2_ckpt.pth"
    trainer.save_checkpoint(str(path))

    model2 = DummyModel()
    optimizer2 = optim.SGD(model2.parameters(), lr=0.1)
    trainer2 = TrainerV2(
        model2,
        optimizer2,
        device,
        compute_loss=compute_loss,
        total_epochs=2,
        ema_decay=0.9,
    )
    trainer2.load_checkpoint(str(path))

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def main():
    test_lr_scheduler()
    print("LR scheduler test passed")
    test_ema_state()
    print("EMA state test passed")
    test_trainer_v2_basic()
    print("TrainerV2 basic test passed")
    test_multiview_trainer()
    print("MultiViewPoseTrainerV2 test passed")
    test_checkpoint_roundtrip()
    print("Checkpoint roundtrip test passed")
    print("All trainer_v2 smoke tests passed")


if __name__ == "__main__":
    main()
