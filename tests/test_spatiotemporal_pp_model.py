"""CPU smoke tests for the factorised (T x V x J) principal-point model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.models.spatiotemporal_principal_point_model import (
    SpatiotemporalPrincipalPointModel,
    _make_cameras,
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = SpatiotemporalPrincipalPointModel(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_compatibility():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = SpatiotemporalPrincipalPointModel(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


def test_per_sample_rig():
    B, T, V, J = 2, 5, 4, 17
    x = torch.rand(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.zeros(V, 3)

    model = SpatiotemporalPrincipalPointModel(j=J, d=64, n_views=V)
    pred, weights = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)


def test_return_pp_delta():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = SpatiotemporalPrincipalPointModel(
        j=J, d=64, n_views=V, return_pp_delta=True
    )
    pred, weights, pp_delta = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B, T, V, 2)


def test_parameter_count():
    J = 17
    model = SpatiotemporalPrincipalPointModel(j=J, d=64, n_views=4)
    total = count_parameters(model)
    # The skeleton is ~1.3 M parameters for these defaults.
    assert total > 0
    print(f"\nSpatiotemporalPrincipalPointModel parameters: {total:,}")


if __name__ == "__main__":
    test_forward_shape_and_grad()
    print("forward + grad test passed")
    test_single_frame_compatibility()
    print("single-frame compatibility test passed")
    test_per_sample_rig()
    print("per-sample rig test passed")
    test_return_pp_delta()
    print("return pp delta test passed")
    test_parameter_count()
    print("all spatiotemporal principal-point tests passed")
