"""Smoke tests for the distilled lightweight student model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.models.distilled_student_principal_point_model import DistilledStudentPrincipalPointModel


def _make_cameras(V: int = 4):
    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    R = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    t = torch.zeros(V, 3).float()
    return K, R, t


def test_distilled_student_forward_backward():
    B, T, V, J = 2, 13, 4, 17
    x = torch.randn(B, T, V, J, 3)
    x[..., 2] = torch.sigmoid(x[..., 2])
    K, R, t = _make_cameras(V)
    model = DistilledStudentPrincipalPointModel(
        j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64, return_pp_delta=True
    )
    pred, weights, pp_delta = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)

    loss = pred.mean() + weights.mean() * 1e-3
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_distilled_student_single_frame():
    V, J = 4, 17
    x = torch.randn(1, V, J, 3)
    x[..., 2] = torch.sigmoid(x[..., 2])
    K, R, t = _make_cameras(V)
    model = DistilledStudentPrincipalPointModel(
        j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64
    )
    pred, weights = model(x, K=K, R=R, t=t)
    assert pred.shape == (1, J, 3)
    assert weights.shape == (1, V, J)


def test_distilled_student_param_count():
    student = DistilledStudentPrincipalPointModel(
        j=17, d=32, n_views=4, n_st_layers=1, residual_hidden=64, return_pp_delta=True
    )
    assert sum(p.numel() for p in student.parameters()) < 500_000
