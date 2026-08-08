"""CPU tests for the v26 differentiable camera calibration refinement module."""

import pytest
import torch

from motionflow_mv.calibration.camera_refinement_v26 import CameraRefinementV26, _reprojection_loss


@pytest.fixture
def synthetic_scene():
    """Return a simple multi-view scene with known ground-truth cameras."""
    B, T, V, J = 2, 3, 4, 17
    # 3-D skeleton roughly 3 m in front of the cameras.
    X = torch.randn(B, T, J, 3) * 0.3
    X[..., 2] = X[..., 2].abs() + 2.5

    # Pinhole cameras arranged in a quarter-circle.
    K = torch.zeros(B, T, V, 3, 3)
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0
    K[..., 2, 2] = 1.0

    angles = torch.linspace(0, 1.5, V)
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3).clone()
    t = torch.zeros(B, T, V, 3)
    for i, a in enumerate(angles):
        ca, sa = a.cos(), a.sin()
        # Camera looks toward origin from a position on the arc.
        cam_pos = torch.tensor([sa * 4.0, 0.0, ca * 4.0])
        forward = -cam_pos / cam_pos.norm()
        world_up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.cross(forward, world_up, dim=-1)
        right = right / right.norm()
        up = torch.cross(right, forward, dim=-1)
        R[:, :, i] = torch.stack([right, up, forward], dim=0)
        t[:, :, i] = -torch.matmul(R[:, :, i], cam_pos)

    # Project to 2-D.
    X_cam = torch.matmul(R.unsqueeze(3), X.unsqueeze(2).unsqueeze(-1)).squeeze(-1) + t.unsqueeze(3)
    proj = torch.matmul(K.unsqueeze(3), X_cam.unsqueeze(-1)).squeeze(-1)
    points_2d = proj[..., :2] / proj[..., 2:3]

    weights = torch.ones(B, T, V, J)
    return {
        "points_2d": points_2d,
        "X": X,
        "K": K,
        "R": R,
        "t": t,
        "weights": weights,
    }


def test_reprojection_loss_positive(synthetic_scene):
    loss = _reprojection_loss(
        synthetic_scene["X"],
        synthetic_scene["points_2d"],
        synthetic_scene["K"],
        synthetic_scene["R"],
        synthetic_scene["t"],
        synthetic_scene["weights"],
    )
    assert loss.item() >= 0.0


def test_identity_at_init(synthetic_scene):
    module = CameraRefinementV26(n_steps=2, lr=0.05)
    K, R, t = synthetic_scene["K"], synthetic_scene["R"], synthetic_scene["t"]
    K_ref, R_ref, t_ref = module(
        synthetic_scene["points_2d"],
        synthetic_scene["X"],
        K,
        R,
        t,
        synthetic_scene["weights"],
    )
    assert K_ref.shape == K.shape
    assert R_ref.shape == R.shape
    assert t_ref.shape == t.shape
    # Gate is zero => input and output are essentially identical.
    assert torch.allclose(K_ref, K, atol=1e-5)
    assert torch.allclose(R_ref, R, atol=1e-5)
    assert torch.allclose(t_ref, t, atol=1e-5)


def test_gate_opens_and_changes_cameras(synthetic_scene):
    module = CameraRefinementV26(n_steps=3, lr=0.1)
    # Force the gate open.
    module.residual_scale.data.fill_(5.0)
    K, R, t = synthetic_scene["K"], synthetic_scene["R"], synthetic_scene["t"]
    K_ref, R_ref, t_ref = module(
        synthetic_scene["points_2d"],
        synthetic_scene["X"],
        K,
        R,
        t,
        synthetic_scene["weights"],
    )
    # Cameras should have moved (at least one of K/R/t).
    assert not torch.allclose(K_ref, K, atol=1e-6) or not torch.allclose(t_ref, t, atol=1e-6)


def test_gate_is_learnable(synthetic_scene):
    module = CameraRefinementV26(n_steps=2, lr=0.1)
    module.residual_scale.data.fill_(3.0)
    K, R, t = synthetic_scene["K"], synthetic_scene["R"], synthetic_scene["t"]

    K_ref, R_ref, t_ref = module(
        synthetic_scene["points_2d"],
        synthetic_scene["X"],
        K,
        R,
        t,
        synthetic_scene["weights"],
    )
    loss = K_ref.mean()
    loss.backward()
    assert module.residual_scale.grad is not None
    assert not torch.isnan(module.residual_scale.grad)


def test_reprojection_error_decreases_when_gate_open(synthetic_scene):
    from motionflow_mv.calibration.perturb import perturb_cameras

    module = CameraRefinementV26(n_steps=5, lr=0.1)
    module.residual_scale.data.fill_(3.0)
    K, R, t = synthetic_scene["K"], synthetic_scene["R"], synthetic_scene["t"]

    # Perturb the ground-truth cameras so that there is something to refine.
    K_pert, R_pert, t_pert = perturb_cameras(
        K, R, t, rot_std=2.0, trans_std=0.10, focal_std=0.03, pp_std=5.0
    )

    initial_loss = _reprojection_loss(
        synthetic_scene["X"],
        synthetic_scene["points_2d"],
        K_pert,
        R_pert,
        t_pert,
        synthetic_scene["weights"],
    ).item()

    K_ref, R_ref, t_ref = module(
        synthetic_scene["points_2d"],
        synthetic_scene["X"],
        K_pert,
        R_pert,
        t_pert,
        synthetic_scene["weights"],
    )

    refined_loss = _reprojection_loss(
        synthetic_scene["X"],
        synthetic_scene["points_2d"],
        K_ref.detach(),
        R_ref.detach(),
        t_ref.detach(),
        synthetic_scene["weights"],
    ).item()

    assert refined_loss <= initial_loss + 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
