import pytest
import torch

from motionflow_mv.fusion.self_evolution_feedback_head_v50 import (
    SelfEvolutionFeedbackHeadV50,
    compute_sefh_loss,
)


def test_self_evolution_feedback_head_forward():
    B, T, V, J = 2, 4, 3, 17
    pred_3d = torch.randn(B, T, J, 3)
    points_2d = torch.randn(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3).clone()
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3).clone()
    t = torch.randn(B, V, 3) * 0.1
    view_mask = torch.ones(B, T, V)

    sefh = SelfEvolutionFeedbackHeadV50(j=J, hidden=16, num_layers=2)
    reliability, log_var, reproj, temporal, epipolar, _ = sefh(
        pred_3d, points_2d, K, R, t, view_mask=view_mask
    )
    assert reliability.shape == (B, T, V, J)
    assert log_var.shape == (B, T, J)
    assert reproj.shape == (B, T, V, J)
    assert temporal.shape == (B, T, J)
    assert epipolar.shape == (B, T, V, J)
    assert reliability.min() >= 0.05
    assert reliability.max() <= 1.0


def test_compute_sefh_loss():
    B, T, V, J = 2, 4, 3, 17
    reliability = torch.rand(B, T, V, J)
    log_var = torch.randn(B, T, J)
    reproj = torch.rand(B, T, V, J)
    temporal = torch.rand(B, T, J)
    epipolar = torch.rand(B, T, V, J)
    view_mask = torch.ones(B, T, V)
    loss = compute_sefh_loss(reliability, log_var, reproj, temporal, epipolar, view_mask)
    assert loss.numel() == 1 and loss.item() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
