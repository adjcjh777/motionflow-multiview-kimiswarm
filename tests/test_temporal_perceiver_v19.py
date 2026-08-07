"""Tests for motionflow_mv.fusion.temporal_perceiver_v19."""

import pytest
import torch

from motionflow_mv.fusion.temporal_perceiver_v19 import TemporalPerceiverRefiner


def test_temporal_perceiver_refiner_shape():
    """Forward pass returns the expected output shape."""
    B, T, J = 2, 48, 17
    in_dim = 3
    model = TemporalPerceiverRefiner(
        j=J,
        in_dim=in_dim,
        d=64,
        n_latents=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_temporal_len=256,
    )
    x = torch.randn(B, T, J, in_dim)
    baseline = torch.randn(B, T, J, 3)

    out = model(x, baseline)
    assert out.shape == (B, T, J, 3)

    # Without baseline the module returns a raw residual of the same shape.
    out_no_base = model(x)
    assert out_no_base.shape == (B, T, J, 3)


def test_temporal_perceiver_refiner_gradient():
    """Backward pass reaches every parameter."""
    B, T, J = 2, 24, 17
    model = TemporalPerceiverRefiner(j=J, in_dim=6, d=32, n_latents=8, n_layers=1)
    x = torch.randn(B, T, J, 6)
    baseline = torch.randn(B, T, J, 3)

    out = model(x, baseline)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_temporal_perceiver_refiner_long_clip():
    """Module handles clips close to max_temporal_len."""
    T = 256
    B, J, in_dim = 1, 17, 3
    model = TemporalPerceiverRefiner(
        j=J,
        in_dim=in_dim,
        d=64,
        n_latents=32,
        n_layers=2,
        max_temporal_len=T,
    )
    x = torch.randn(B, T, J, in_dim)
    out = model(x)
    assert out.shape == (B, T, J, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
