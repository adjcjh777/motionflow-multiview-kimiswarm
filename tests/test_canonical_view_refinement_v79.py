"""Unit tests for the v79 canonical-view geometric refinement head."""

from __future__ import annotations

import pytest
import torch

from motionflow_mv.fusion.canonical_view_refinement_v79 import (
    CanonicalViewRefinementV79,
)


@pytest.fixture
def random_inputs():
    """Return a small deterministic set of random inputs."""
    torch.manual_seed(0)
    B, T, V, J = 2, 3, 4, 17
    pred_3d = torch.randn(B, T, J, 3) * 0.5
    K = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3).clone()
    # Make intrinsics slightly non-trivial.
    K[..., 0, 0] += 100.0
    K[..., 1, 1] += 100.0
    K[..., 0, 2] += 320.0
    K[..., 1, 2] += 240.0
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3).clone()
    t = torch.randn(B, T, V, 3) * 2.0
    return pred_3d, K, R, t


def test_output_shape_matches_input(random_inputs):
    """The refined pose must have the same shape as the input pose."""
    pred_3d, K, R, t = random_inputs
    model = CanonicalViewRefinementV79(j=pred_3d.shape[-2])
    out = model(pred_3d, K, R, t)
    assert out.shape == pred_3d.shape


def test_identity_at_init(random_inputs):
    """With identity init and a closed gate the output should equal the input."""
    pred_3d, K, R, t = random_inputs
    model = CanonicalViewRefinementV79(
        j=pred_3d.shape[-2],
        identity_init=True,
        residual_gate_init=-6.0,
    )
    out = model(pred_3d, K, R, t)
    # The gate is ~0.0025, the residual is zero; output should be the input.
    assert torch.allclose(out, pred_3d, atol=1e-5)


def test_gradients_flow(random_inputs):
    """A loss on the output must back-propagate to the module parameters."""
    pred_3d, K, R, t = random_inputs
    model = CanonicalViewRefinementV79(j=pred_3d.shape[-2])
    out = model(pred_3d, K, R, t)
    loss = out.sum()
    loss.backward()

    assert model.gate_logit.grad is not None
    assert model.residual_out.weight.grad is not None
    assert model.residual_out.bias.grad is not None
    assert model.first.weight.grad is not None
