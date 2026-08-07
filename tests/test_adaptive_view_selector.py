"""CPU smoke / unit tests for AdaptiveViewSelector.

This module does not start any GPU training; it only checks that the selector
produces valid masks, exact top-k inference masks, and differentiable budget
losses.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.adaptive_view_selector import AdaptiveViewSelector


@pytest.fixture
def selector():
    return AdaptiveViewSelector(
        d=64,
        n_views=4,
        n_joints=17,
        target_k=2,
        temperature=0.5,
        budget_weight=0.1,
    )


def test_training_forward_shape_and_budget(selector: AdaptiveViewSelector):
    B, V, J, d = 4, 4, 17, 64
    x = torch.rand(B, V, J, d)
    selector.train()
    mask, budget_loss = selector(x)

    assert mask.shape == (B, V, J)
    assert budget_loss.numel() == 1
    assert budget_loss.item() >= 0.0
    # Straight-through hard top-k: each joint has exactly target_k selected views.
    assert torch.allclose(mask.sum(dim=1), torch.full((B, J), float(selector.target_k)), atol=1e-5)


def test_inference_exact_topk(selector: AdaptiveViewSelector):
    B, V, J, d = 4, 4, 17, 64
    x = torch.rand(B, V, J, d)
    selector.eval()
    with torch.no_grad():
        mask, budget_loss = selector(x)

    assert mask.shape == (B, V, J)
    assert torch.allclose(mask.sum(dim=1), torch.full((B, J), float(selector.target_k)), atol=1e-5)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})
    assert budget_loss.item() == 0.0


def test_bypass_mode():
    B, V, J, d = 2, 4, 17, 64
    selector = AdaptiveViewSelector(
        d=d,
        n_views=V,
        n_joints=J,
        target_k=2,
        use_selector=False,
    )
    x = torch.rand(B, V, J, d)
    for training in (True, False):
        selector.train(training)
        mask, budget_loss = selector(x)
        assert torch.allclose(mask, torch.ones(B, V, J))
        assert budget_loss.item() == 0.0


def test_target_k_ratio():
    selector = AdaptiveViewSelector(
        d=32,
        n_views=4,
        n_joints=17,
        target_k=0.75,
    )
    assert selector.target_k == 3  # round(0.75 * 4)


def test_gradient_flow(selector: AdaptiveViewSelector):
    B, V, J, d = 2, 4, 17, 64
    x = torch.rand(B, V, J, d, requires_grad=True)
    selector.train()
    mask, budget_loss = selector(x)
    loss = mask.mean() + budget_loss
    loss.backward()

    assert any(p.grad is not None for p in selector.parameters())
    assert x.grad is not None


def test_mean_k_approaches_target():
    """Training samples should average around the target budget."""
    selector = AdaptiveViewSelector(
        d=64,
        n_views=4,
        n_joints=17,
        target_k=3,
        temperature=0.5,
        budget_weight=0.0,
    )
    B, V, J, d = 16, 4, 17, 64
    x = torch.rand(B, V, J, d)
    selector.train()
    with torch.no_grad():
        mask, _ = selector(x)
    mean_k = mask.sum(dim=1).mean().item()
    assert abs(mean_k - selector.target_k) < 1e-4


if __name__ == "__main__":
    test_training_forward_shape_and_budget(selector())
    print("test_training_forward_shape_and_budget passed")
    test_inference_exact_topk(selector())
    print("test_inference_exact_topk passed")
    test_bypass_mode()
    print("test_bypass_mode passed")
    test_target_k_ratio()
    print("test_target_k_ratio passed")
    test_gradient_flow(selector())
    print("test_gradient_flow passed")
    test_mean_k_approaches_target()
    print("test_mean_k_approaches_target passed")
    print("All adaptive view selector tests passed")
