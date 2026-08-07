"""CPU smoke test for the OmniMultiViewFusionV2 trainer.

Imports the trainer defined in ``experiments/train_omniview_fusion_v2_mpiinf3dhp.py``,
runs a single forward/backward step on a tiny synthetic dataset, and checks that
losses and gradients are finite.
"""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.train_omniview_fusion_v2_mpiinf3dhp import (
    OmniMultiViewTrainer,
    SyntheticSmokeDataset,
    _make_synthetic_cameras,
    collate_fn,
)
from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2


def test_trainer_forward_backward():
    """One training step with the OmniMultiView trainer should produce finite gradients."""
    device = torch.device("cpu")
    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17
    clip_len = 9

    dataset = SyntheticSmokeDataset(
        K, R, t, n_frames=32, n_joints=n_joints, clip_len=clip_len
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    model = OmniMultiViewFusionV2(
        j=n_joints,
        d=32,
        n_views=4,
        n_st_layers=1,
        residual_hidden=64,
        graph_num_layers=1,
        return_pp_delta=False,
        return_covariance=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    args = Namespace(
        j=n_joints,
        n_views=4,
        smoke=True,
        noise_std=0.0,
        confidence_dropout=0.0,
        view_dropout_rate=0.0,
        min_views=2,
        visibility_loss_weight=0.1,
        uncertainty_loss_weight=0.05,
        temporal_loss_weight=0.02,
        bone_loss_weight=0.05,
    )

    trainer = OmniMultiViewTrainer(
        model,
        optimizer,
        device,
        args=args,
        total_epochs=1,
        max_grad_norm=1.0,
        amp_enabled=False,
        ema_decay=None,
    )

    batch = next(iter(loader))
    metrics = trainer.train_step(batch)

    assert "loss" in metrics
    assert metrics["loss"] < float("inf")
    assert metrics["loss"] == metrics["loss"]  # not NaN
    assert any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
