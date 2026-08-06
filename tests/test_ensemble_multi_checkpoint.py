"""CPU smoke test for multi-checkpoint ensemble inference."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.prototypes.ensemble_predictor import (
    MultiCheckpointEnsemble,
    load_ensemble_from_checkpoints,
)


class _TinyModel(nn.Module):
    """Deterministic tiny model whose output is the sum of a weight and input."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, x):
        return self.weight * x


def _make_checkpoint(tmp_path: Path, weight: float, filename: str):
    model = _TinyModel(weight)
    ckpt = tmp_path / filename
    torch.save(model.state_dict(), ckpt)
    return ckpt


def test_load_ensemble_from_checkpoints(tmp_path):
    ckpt_a = _make_checkpoint(tmp_path, 1.0, "a.pth")
    ckpt_b = _make_checkpoint(tmp_path, 3.0, "b.pth")

    ensemble = load_ensemble_from_checkpoints(
        build_fn=_TinyModel,
        checkpoint_paths=[str(ckpt_a), str(ckpt_b)],
        device="cpu",
    )

    x = torch.ones(2, 4, 3)
    y = ensemble(x)
    expected = 2.0 * torch.ones_like(x)  # mean of 1*x and 3*x
    assert torch.allclose(y, expected)


def test_weighted_ensemble(tmp_path):
    ckpt_a = _make_checkpoint(tmp_path, 1.0, "a.pth")
    ckpt_b = _make_checkpoint(tmp_path, 3.0, "b.pth")

    ensemble = MultiCheckpointEnsemble(
        build_fn=_TinyModel,
        checkpoint_paths=[str(ckpt_a), str(ckpt_b)],
        device="cpu",
        weights=[1.0, 3.0],
    )

    x = torch.ones(2, 4, 3)
    y = ensemble(x)
    # (1*1 + 3*3) / (1+3) = 10/4 = 2.5
    expected = 2.5 * torch.ones_like(x)
    assert torch.allclose(y, expected)


def test_predict_all_returns_both_outputs(tmp_path):
    ckpt = _make_checkpoint(tmp_path, 2.0, "single.pth")
    ensemble = load_ensemble_from_checkpoints(
        build_fn=_TinyModel,
        checkpoint_paths=[str(ckpt)],
        device="cpu",
    )

    x = torch.arange(6).view(2, 3).float()
    mean, individual = ensemble.predict_all(x)
    assert torch.allclose(mean, 2.0 * x)
    assert individual.shape[0] == 1
    assert torch.allclose(individual.squeeze(0), 2.0 * x)


def test_empty_checkpoint_list_raises():
    try:
        MultiCheckpointEnsemble(build_fn=_TinyModel, checkpoint_paths=[])
    except ValueError as exc:
        assert "At least one checkpoint" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty checkpoint list")
