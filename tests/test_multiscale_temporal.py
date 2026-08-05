"""Lightweight sanity tests for the multi-scale temporal convolution model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.multiscale_temporal_conv_model import MultiScaleTemporalConvModel
from motionflow_mv.fusion.ray_attention_temporal_model import _make_cameras


def test_multiscale_temporal_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = MultiScaleTemporalConvModel(j=J, d=64, n_views=V, n_temporal_layers=2)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_multiscale_temporal_single_frame_compatibility():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = MultiScaleTemporalConvModel(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


def test_custom_kernel_dilation():
    B, T, V, J = 1, 7, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = MultiScaleTemporalConvModel(
        j=J, d=64, n_views=V, n_temporal_layers=1,
        temporal_kernel_sizes=[3, 5], temporal_dilations=[1, 2],
    )
    pred, weights = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)


if __name__ == "__main__":
    test_multiscale_temporal_forward_shape_and_grad()
    test_multiscale_temporal_single_frame_compatibility()
    test_custom_kernel_dilation()
    print("multi-scale temporal conv tests passed")
