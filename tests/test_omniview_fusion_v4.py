"""CPU pytest suite for OmniMultiViewFusionV4.

Covers shape/arity, gradient flow, warm-start from a synthetic v2/v3 checkpoint,
all v4 toggles, variable-view inference, and the attention-entropy loss.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from motionflow_mv.fusion.attention_entropy_loss import AttentionEntropyLoss
from motionflow_mv.fusion.omniview_fusion_v3 import OmniMultiViewFusionV3
from motionflow_mv.fusion.omniview_fusion_v4 import (
    OmniMultiViewFusionV4,
    _make_cameras,
)
from motionflow_mv.fusion.variable_view_inference import (
    VariableViewInferenceWrapper,
)


@pytest.fixture
def input_shape():
    return {"B": 2, "T": 9, "V": 4, "J": 17}


@pytest.fixture
def cameras(input_shape):
    return _make_cameras(input_shape["V"])


@pytest.fixture
def x(input_shape):
    return torch.rand(input_shape["B"], input_shape["T"], input_shape["V"], input_shape["J"], 3)


def test_v4_shape_and_arity(input_shape, cameras, x):
    """Default v4 forward returns the expected 5-tuple."""
    B, T, V, J = input_shape["B"], input_shape["T"], input_shape["V"], input_shape["J"]
    model = OmniMultiViewFusionV4(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )
    pred, weights, visibility, covariance, epi_loss = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)
    assert covariance.shape == (B, T, V, J, 2, 2)
    assert epi_loss.numel() == 1
    assert torch.isfinite(epi_loss)


def test_v4_gradient_flow(input_shape, cameras, x):
    """Gradients flow through the full default v4 model on CPU."""
    T, V, J = input_shape["T"], input_shape["V"], input_shape["J"]
    model = OmniMultiViewFusionV4(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )
    pred, _, _, _, epi_loss = model(x, cameras=cameras)
    loss = pred.mean() + epi_loss
    loss.backward()

    assert any(p.grad is not None for p in model.parameters())


def test_v4_warm_start_from_v3_checkpoint(input_shape, cameras, x):
    """v4 loads a synthetic v3 checkpoint with strict=False."""
    B, T, V, J = input_shape["B"], input_shape["T"], input_shape["V"], input_shape["J"]
    v3 = OmniMultiViewFusionV3(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
    )
    v3.eval()
    with torch.no_grad():
        checkpoint_buffer = io.BytesIO()
        torch.save(v3.state_dict(), checkpoint_buffer)
        checkpoint_buffer.seek(0)
        state_dict = torch.load(checkpoint_buffer, map_location="cpu", weights_only=True)

    v4 = OmniMultiViewFusionV4(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_context_visibility=True,
        use_kinematic_refiner=True,
    )
    missing, unexpected = v4.load_state_dict(state_dict, strict=False)

    # No leftover keys from the v3 checkpoint.
    assert len(unexpected) == 0
    # v4-specific modules are absent from the v3 checkpoint, so they are missing.
    assert len(missing) > 0

    # Loaded model still runs.
    v4.eval()
    with torch.no_grad():
        out = v4(x, cameras=cameras)
    assert out[0].shape == (B, T, J, 3)


def test_v4_all_toggles_on(input_shape, cameras, x):
    """All v4 toggles enabled still run and produce gradients."""
    T, V, J = input_shape["T"], input_shape["V"], input_shape["J"]
    model = OmniMultiViewFusionV4(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_context_visibility=True,
        use_skeleton_residual=True,
        use_kinematic_refiner=True,
        use_adaptive_view_selection=True,
        use_rotation_correction=True,
        use_entropy_regularization=True,
    )
    pred, weights, visibility, covariance, epi_loss = model(x, cameras=cameras)

    assert pred.shape == (input_shape["B"], T, J, 3)
    assert weights.shape == (input_shape["B"], T, V, J)
    assert visibility.shape == (input_shape["B"], T, V, J)
    assert covariance.shape == (input_shape["B"], T, V, J, 2, 2)
    assert epi_loss.numel() == 1
    assert torch.isfinite(epi_loss)

    (pred.mean() + epi_loss).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_v4_toggles_off_match_v3_shape(input_shape, cameras, x):
    """v4 with all new toggles off matches the v3 output arity."""
    B, T, V, J = input_shape["B"], input_shape["T"], input_shape["V"], input_shape["J"]
    v4 = OmniMultiViewFusionV4(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )
    v3 = OmniMultiViewFusionV3(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )
    with torch.no_grad():
        out_v4 = v4(x, cameras=cameras)
        out_v3 = v3(x, cameras=cameras)

    assert len(out_v4) == len(out_v3)
    for a, b in zip(out_v4, out_v3):
        assert a.shape == b.shape


def test_v4_variable_view_wrapper(input_shape):
    """VariableViewInferenceWrapper can run v4 with 2 active views."""
    B, T, V, J = input_shape["B"], input_shape["T"], input_shape["V"], input_shape["J"]
    model = OmniMultiViewFusionV4(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
    )
    wrapper = VariableViewInferenceWrapper(model)

    # Synthetic input with all 4 views.
    x = torch.rand(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)
    t = torch.zeros(B, V, 3)

    out = wrapper(x, K=K, R=R, t=t, active_views=2)
    pred = out[0]
    assert pred.shape == (B, T, J, 3)
    assert torch.isfinite(pred).all()


def test_v4_entropy_loss_matches_expectation():
    """AttentionEntropyLoss is non-negative, zero for one-hot, differentiable."""
    B, T, V, J = 2, 3, 4, 17
    loss_fn = AttentionEntropyLoss(weight=0.01, dim=-2)

    # One-hot weights -> zero loss.
    one_hot = torch.zeros(B, T, V, J)
    one_hot[:, :, 0, :] = 1.0
    loss_one_hot = loss_fn(one_hot)
    assert loss_one_hot.item() == pytest.approx(0.0, abs=1e-5)

    # Uniform weights -> positive loss.
    uniform = torch.ones(B, T, V, J)
    loss_uniform = loss_fn(uniform)
    assert loss_uniform.item() > 0.0

    # Differentiability.
    weights = torch.rand(B, T, V, J, requires_grad=True)
    loss = loss_fn(weights)
    loss.backward()
    assert weights.grad is not None
    assert weights.grad.shape == weights.shape
