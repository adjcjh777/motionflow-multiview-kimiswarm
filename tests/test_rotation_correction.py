"""Lightweight sanity tests for the rotation correction head."""

import math

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.rotation_correction import (
    RotationCorrectionHead,
    _geodesic_angle,
)


@pytest.fixture
def toy_setup():
    N, V, d = 2, 4, 64
    feat = torch.randn(N, V, d)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(N, V, 3, 3).clone()
    return N, V, d, feat, R


def test_rotation_correction_shape_and_no_nan(toy_setup):
    N, V, d, feat, R = toy_setup
    head = RotationCorrectionHead(d=d, hidden=64, max_rot_deg=2.0)
    R_corrected, delta_R = head(feat, R)

    assert R_corrected.shape == (N, V, 3, 3)
    assert delta_R.shape == (N, V, 3, 3)
    assert torch.isfinite(R_corrected).all()
    assert torch.isfinite(delta_R).all()


def test_rotation_correction_identity_at_init(toy_setup):
    _, _, _, feat, R = toy_setup
    head = RotationCorrectionHead(d=64, hidden=64, max_rot_deg=5.0)
    R_corrected, _ = head(feat, R)
    assert torch.allclose(R_corrected, R, atol=1e-6)


def test_rotation_correction_so3_validity(toy_setup):
    _, V, _, feat, R = toy_setup
    head = RotationCorrectionHead(d=64, hidden=64, max_rot_deg=5.0)
    _, delta_R = head(feat, R)

    I = torch.eye(3, device=delta_R.device, dtype=delta_R.dtype)
    identity_check = torch.einsum("nvij,nvjk->nvik", delta_R, delta_R.transpose(-2, -1))
    assert torch.allclose(identity_check, I, atol=1e-4)
    assert ((torch.det(delta_R) - 1.0).abs() < 1e-4).all()


def test_rotation_correction_bounds_known_rotation(toy_setup):
    _, V, _, feat, R = toy_setup
    head = RotationCorrectionHead(d=64, hidden=64, max_rot_deg=2.0)
    with torch.no_grad():
        head.mlp[-1].bias[0] = 0.5
    R_corrected, delta_R = head(feat, R)

    angle = _geodesic_angle(delta_R)
    max_angle = math.radians(2.0)
    assert (angle <= max_angle + 1e-5).all()
    assert not torch.allclose(R_corrected, R, atol=1e-6)


def test_rotation_correction_gradient_flow(toy_setup):
    _, _, _, feat, R = toy_setup
    head = RotationCorrectionHead(d=64, hidden=64, max_rot_deg=2.0)
    R_corrected, _ = head(feat, R)
    loss = R_corrected.sum()
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_rotation_correction_pools_per_joint_features(toy_setup):
    N, V, d, _, R = toy_setup
    feat_4d = torch.randn(N, V, 17, d)
    head = RotationCorrectionHead(d=d, hidden=64, max_rot_deg=2.0)
    R_corrected, _ = head(feat_4d, R)
    assert R_corrected.shape == (N, V, 3, 3)


def test_geodesic_angle_identity():
    R = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
    angles = _geodesic_angle(R)
    assert torch.allclose(angles, torch.zeros_like(angles), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
