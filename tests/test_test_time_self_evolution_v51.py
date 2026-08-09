"""Unit tests for v51 Test-Time Self-Evolution Refiner."""

from __future__ import annotations

import pytest
import torch

from motionflow_mv.fusion.test_time_self_evolution_v51 import (
    TestTimeSelfEvolutionRefinerV51,
)


@pytest.fixture
def shapes():
    B, T, V, J = 2, 5, 4, 17
    return B, T, V, J


def test_forward_shape(shapes):
    B, T, V, J = shapes
    module = TestTimeSelfEvolutionRefinerV51(n_views=V, n_joints=J, num_steps=2)

    pose = torch.randn(B, T, J, 3)
    x_2d = torch.randn(B, T, V, J, 2)
    K = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    R = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    t = torch.randn(B, V, 3)

    refined_pose, rel, unc = module(pose, x_2d, K, R, t)
    assert refined_pose.shape == pose.shape
    assert rel.shape == (B, V)
    assert unc.shape == (B, J)
    assert (rel >= 0.05).all() and (rel <= 1.0).all()
    assert (unc > 0).all()


def test_identity_with_zero_steps(shapes):
    B, T, V, J = shapes
    module = TestTimeSelfEvolutionRefinerV51(n_views=V, n_joints=J, num_steps=0)

    pose = torch.randn(B, T, J, 3)
    x_2d = torch.randn(B, T, V, J, 2)
    K = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    R = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    t = torch.randn(B, V, 3)

    refined_pose, rel, unc = module(pose, x_2d, K, R, t)
    torch.testing.assert_close(refined_pose, pose)
    assert rel.shape == (B, V)
    assert unc.shape == (B, J)


def test_sefh_seed(shapes):
    B, T, V, J = shapes
    module = TestTimeSelfEvolutionRefinerV51(n_views=V, n_joints=J, num_steps=0)

    pose = torch.randn(B, T, J, 3)
    x_2d = torch.randn(B, T, V, J, 2)
    K = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    R = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    t = torch.randn(B, V, 3)
    sefh_rel = torch.rand(B, V)
    sefh_log_var = torch.randn(B, J)

    _, rel, unc = module(pose, x_2d, K, R, t, sefh_reliability=sefh_rel, sefh_log_var=sefh_log_var)
    assert rel.shape == (B, V)
    assert unc.shape == (B, J)


def test_single_frame(shapes):
    B, V, J = 2, 4, 17
    module = TestTimeSelfEvolutionRefinerV51(n_views=V, n_joints=J, num_steps=2)

    pose = torch.randn(B, 1, J, 3)
    x_2d = torch.randn(B, 1, V, J, 2)
    K = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    R = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    t = torch.randn(B, V, 3)

    refined_pose, rel, unc = module(pose, x_2d, K, R, t)
    assert refined_pose.shape == (B, 1, J, 3)
    assert rel.shape == (B, V)
    assert unc.shape == (B, J)


def test_refine_pose_changes_output(shapes):
    B, T, V, J = shapes
    pose = torch.randn(B, T, J, 3)
    x_2d = torch.randn(B, T, V, J, 2)
    K = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    R = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    t = torch.randn(B, V, 3)

    module_no_refine = TestTimeSelfEvolutionRefinerV51(
        n_views=V, n_joints=J, num_steps=2, refine_pose=False
    )
    module_refine = TestTimeSelfEvolutionRefinerV51(
        n_views=V, n_joints=J, num_steps=2, refine_pose=True
    )

    refined_no_refine, _, _ = module_no_refine(pose, x_2d, K, R, t)
    refined_refine, _, _ = module_refine(pose, x_2d, K, R, t)

    torch.testing.assert_close(refined_no_refine, pose)
    assert not torch.allclose(refined_refine, pose, atol=1e-6)
