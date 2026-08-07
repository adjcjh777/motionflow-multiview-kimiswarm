"""Unit tests for motionflow_mv/fusion/smpl_prior_fusion_v22.py."""

import pytest
import torch

from motionflow_mv.fusion.smpl_prior_fusion_v22 import (
    SMPLPriorHead,
    SMPLPriorFusionV22,
    HAS_SMPLX,
)


BATCH = 2
TIME = 3
VIEWS = 4
JOINTS = 17
D = 64


def _make_cameras(n_views: int = VIEWS):
    """Return a list of simple pinhole cameras for testing."""
    import numpy as np
    from motionflow_mv.calibration.camera import Camera

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def test_smpl_prior_head_output_shapes():
    """The SMPL head must produce correctly shaped parameter predictions."""
    head = SMPLPriorHead(d=D, n_joints=JOINTS)
    residual_input = torch.randn(BATCH * TIME, JOINTS, D + 3)
    out = head(residual_input)

    assert out["betas"].shape == (1, 10)
    assert out["body_pose"].shape == (BATCH * TIME, 69)
    assert out["global_orient"].shape == (BATCH * TIME, 3)
    assert out["transl"].shape == (BATCH * TIME, 3)
    assert out["blend"].shape == (BATCH * TIME, 1)
    assert 0.0 <= out["blend"].min().item() <= out["blend"].max().item() <= 1.0


def test_smpl_prior_fusion_v22_forward():
    """The full fusion model must return the expected output tuple."""
    cameras = _make_cameras()
    x = torch.rand(BATCH, TIME, VIEWS, JOINTS, 3)
    y_3d = torch.randn(BATCH, TIME, JOINTS, 3)

    model = SMPLPriorFusionV22(
        j=JOINTS,
        d=D,
        n_views=VIEWS,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )

    pred_3d, weights, visibility, L, epi_loss = model(x, cameras=cameras)
    assert pred_3d.shape == (BATCH, TIME, JOINTS, 3)
    assert weights.shape == (BATCH, TIME, VIEWS, JOINTS)
    assert visibility.shape == (BATCH, TIME, VIEWS, JOINTS)
    assert L.shape == (BATCH, TIME, VIEWS, JOINTS, 2, 2)
    assert epi_loss.numel() == 1

    loss = (pred_3d - y_3d).pow(2).mean() + epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_smpl_prior_fusion_v22_returns_smpl_dict():
    """When return_smpl=True the model must append the SMPL output dictionary."""
    cameras = _make_cameras()
    x = torch.rand(BATCH, TIME, VIEWS, JOINTS, 3)

    model = SMPLPriorFusionV22(
        j=JOINTS,
        d=D,
        n_views=VIEWS,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )

    out = model(x, cameras=cameras, return_smpl=True)
    assert isinstance(out, tuple)
    # Base tuple has 5 elements; last is the SMPL dict.
    assert len(out) == 6
    smpl_out = out[-1]
    assert smpl_out is not None
    assert smpl_out["betas"].shape == (1, 10)
    assert smpl_out["body_pose"].shape == (BATCH * TIME, 69)
    assert "blend" in smpl_out


def test_freeze_base_disables_base_gradients():
    """freeze_base=True should keep only the SMPL head trainable."""
    model = SMPLPriorFusionV22(
        j=JOINTS,
        d=D,
        n_views=VIEWS,
        freeze_base=True,
    )
    base_params = set(p for n, p in model.named_parameters() if "shape_pose_head" not in n)
    head_params = set(p for n, p in model.named_parameters() if "shape_pose_head" in n)

    assert all(not p.requires_grad for p in base_params)
    assert all(p.requires_grad for p in head_params)
