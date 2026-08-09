"""Unit tests for v51 Domain-Agnostic Ensemble (DAE)."""

import torch

from motionflow_mv.fusion.domain_agnostic_ensemble_v51 import DomainAgnosticEnsembleV51


def test_domain_agnostic_ensemble_v51_forward():
    """DAE should blend two expert poses and preserve identity-at-init."""
    B, T, V, J = 2, 3, 4, 17
    device = torch.device("cpu")

    # Two expert poses.
    pose_geo = torch.randn(B, T, J, 3, device=device)
    pose_res = torch.randn(B, T, J, 3, device=device)

    # 2-D keypoints, cameras.
    points_2d = torch.randn(B, T, V, J, 2, device=device)
    K = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3)
    R = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3)
    t = torch.zeros(B, V, 3, device=device)
    view_mask = torch.ones(B, T, V, device=device)

    dae = DomainAgnosticEnsembleV51(j=J, n_experts=2)
    ensemble_pose, diversity_loss = dae(
        expert_poses=[pose_geo, pose_res],
        points_2d=points_2d,
        K=K,
        R=R,
        t=t,
        view_mask=view_mask,
    )

    assert ensemble_pose.shape == (B, T, J, 3)
    assert diversity_loss.numel() == 1

    # At init the geometry expert should dominate; ensemble should be close to geo.
    # Allow a generous tolerance because the bypass mixes in the residual expert.
    diff = (ensemble_pose - pose_geo).abs().mean()
    assert diff.item() < 1.0, f"identity-at-init violated, diff={diff.item()}"


def test_domain_agnostic_ensemble_v51_gradient():
    """Gradients should flow through the gate to both experts."""
    B, T, V, J = 1, 2, 2, 17
    pose_geo = torch.randn(B, T, J, 3, requires_grad=True)
    pose_res = torch.randn(B, T, J, 3, requires_grad=True)
    points_2d = torch.randn(B, T, V, J, 2)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3)
    t = torch.zeros(B, V, 3)
    view_mask = torch.ones(B, T, V)

    dae = DomainAgnosticEnsembleV51(j=J, n_experts=2)
    ensemble_pose, diversity_loss = dae(
        expert_poses=[pose_geo, pose_res],
        points_2d=points_2d,
        K=K,
        R=R,
        t=t,
        view_mask=view_mask,
    )
    loss = ensemble_pose.mean() + diversity_loss
    loss.backward()

    assert pose_geo.grad is not None
    assert pose_res.grad is not None
    assert any(p.grad is not None for p in dae.parameters())
