import torch
import pytest

from motionflow_mv.fusion.uncertainty_depth_proposal_v27 import UncertaintyDepthProposalTriangulation


def test_uncertainty_depth_proposal_shape():
    n_views, n_ray_samples = 4, 8
    head = UncertaintyDepthProposalTriangulation(n_views=n_views, n_ray_samples=n_ray_samples)
    B, T, V, J = 2, 3, n_views, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.rand(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)

    refined, loss = head(centre, direction, confidence, pred_3d)
    assert refined.shape == (B, T, J, 3)
    assert loss.numel() == 1
    assert torch.isfinite(loss)


def test_uncertainty_depth_proposal_identity_at_init():
    """At init residual_scale=0, so refined should equal pred_3d."""
    head = UncertaintyDepthProposalTriangulation(n_views=4, n_ray_samples=4)
    B, T, V, J = 1, 1, 4, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.ones(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)

    refined, _ = head(centre, direction, confidence, pred_3d)
    assert torch.allclose(refined, pred_3d, atol=1e-5)


def test_uncertainty_depth_proposal_view_mask():
    head = UncertaintyDepthProposalTriangulation(n_views=4, n_ray_samples=4)
    B, T, V, J = 2, 1, 4, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.ones(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)
    view_mask = torch.tensor([[[True, True, False, False]], [[True, True, True, False]]])

    refined, _ = head(centre, direction, confidence, pred_3d, view_mask=view_mask)
    assert refined.shape == (B, T, J, 3)


def test_uncertainty_depth_proposal_inference_deterministic():
    """At inference, using mu only should be deterministic."""
    head = UncertaintyDepthProposalTriangulation(n_views=4, n_ray_samples=4)
    head.eval()
    B, T, V, J = 1, 1, 4, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.ones(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)

    with torch.no_grad():
        out1, _ = head(centre, direction, confidence, pred_3d)
        out2, _ = head(centre, direction, confidence, pred_3d)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_uncertainty_depth_proposal_gmm_shape():
    n_views, n_ray_samples, n_mixtures = 4, 8, 3
    head = UncertaintyDepthProposalTriangulation(
        n_views=n_views, n_ray_samples=n_ray_samples, n_mixtures=n_mixtures
    )
    B, T, V, J = 2, 3, n_views, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.rand(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)

    refined, loss = head(centre, direction, confidence, pred_3d)
    assert refined.shape == (B, T, J, 3)
    assert loss.numel() == 1
    assert torch.isfinite(loss)


def test_uncertainty_depth_proposal_gmm_identity_at_init():
    head = UncertaintyDepthProposalTriangulation(n_views=4, n_ray_samples=4, n_mixtures=2)
    B, T, V, J = 1, 1, 4, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.ones(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)

    refined, _ = head(centre, direction, confidence, pred_3d)
    assert torch.allclose(refined, pred_3d, atol=1e-5)


def test_uncertainty_depth_proposal_gmm_inference_deterministic():
    head = UncertaintyDepthProposalTriangulation(n_views=4, n_ray_samples=4, n_mixtures=2)
    head.eval()
    B, T, V, J = 1, 1, 4, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.ones(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)

    with torch.no_grad():
        out1, _ = head(centre, direction, confidence, pred_3d)
        out2, _ = head(centre, direction, confidence, pred_3d)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_uncertainty_depth_proposal_training_stochastic():
    """During training, stochastic sampling should produce different outputs."""
    head = UncertaintyDepthProposalTriangulation(n_views=4, n_ray_samples=8, uncertainty_loss_weight=0.0)
    head.train()
    # Force non-zero residual scale to amplify any difference.
    head.residual_scale.data.fill_(1.0)
    B, T, V, J = 1, 1, 4, 17
    centre = torch.randn(B, T, V, 3)
    direction = torch.randn(B, T, V, J, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    confidence = torch.ones(B, T, V, J)
    pred_3d = torch.randn(B, T, J, 3)

    out1, _ = head(centre, direction, confidence, pred_3d)
    out2, _ = head(centre, direction, confidence, pred_3d)
    # With high probability the outputs differ; tolerance set to allow rare equality.
    assert not torch.allclose(out1, out2, atol=1e-6)
