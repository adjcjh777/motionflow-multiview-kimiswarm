"""Unit tests for the diffusion-based pose refinement head (v20)."""

import pytest
import torch

from motionflow_mv.fusion.diffusion_pose_refiner_v20 import DiffusionPoseRefinerV20


def test_diffusion_refiner_inference_shape():
    """Inference should return a refined pose of the same shape."""
    B, T, J = 2, 5, 17
    pose = torch.randn(B, T, J, 3)
    feat = torch.randn(B * T, 64)

    refiner = DiffusionPoseRefinerV20(
        j=J,
        in_dim=64,
        residual_hidden=32,
        num_diffusion_steps=10,
        num_inference_steps=2,
    )
    out = refiner(pose, feat=feat)
    assert out.shape == (B, T, J, 3)


def test_diffusion_refiner_training_loss_shape():
    """Training forward should return a refined pose and a scalar loss."""
    B, T, J = 3, 4, 17
    pose = torch.randn(B, T, J, 3)
    feat = torch.randn(B * T, 64)
    targets = torch.randn(B, T, J, 3)

    refiner = DiffusionPoseRefinerV20(
        j=J,
        in_dim=64,
        residual_hidden=32,
        num_diffusion_steps=10,
        num_inference_steps=2,
    )
    refined, loss = refiner(pose, feat=feat, train_targets=targets)
    assert refined.shape == (B, T, J, 3)
    assert loss.shape == ()
    assert loss.numel() == 1


def test_diffusion_refiner_no_feature_conditioning():
    """Refiner should work without feature conditioning."""
    B, T, J = 2, 3, 17
    pose = torch.randn(B, T, J, 3)
    targets = torch.randn(B, T, J, 3)

    refiner = DiffusionPoseRefinerV20(
        j=J,
        in_dim=None,
        residual_hidden=16,
        num_diffusion_steps=10,
        num_inference_steps=2,
    )
    refined, loss = refiner(pose, train_targets=targets)
    assert refined.shape == (B, T, J, 3)
    assert loss.numel() == 1


def test_diffusion_refiner_single_frame_shape():
    """Refiner should accept a single-frame (B, J, 3) input."""
    B, J = 4, 17
    pose = torch.randn(B, J, 3)
    feat = torch.randn(B, 64)

    refiner = DiffusionPoseRefinerV20(
        j=J,
        in_dim=64,
        residual_hidden=16,
        num_diffusion_steps=8,
        num_inference_steps=2,
    )
    out = refiner(pose, feat=feat)
    assert out.shape == (B, J, 3)


def test_diffusion_refiner_backward():
    """Gradients should flow through the refiner during training."""
    B, T, J = 2, 2, 17
    pose = torch.randn(B, T, J, 3)
    feat = torch.randn(B * T, 64)
    targets = torch.randn(B, T, J, 3)

    refiner = DiffusionPoseRefinerV20(
        j=J,
        in_dim=64,
        residual_hidden=16,
        num_diffusion_steps=8,
        num_inference_steps=2,
    )
    refined, loss = refiner(pose, feat=feat, train_targets=targets)
    loss.backward()
    assert any(p.grad is not None for p in refiner.parameters())


def test_diffusion_refiner_cosine_schedule():
    """Cosine schedule should be selectable."""
    B, T, J = 2, 2, 17
    pose = torch.randn(B, T, J, 3)
    feat = torch.randn(B * T, 64)

    refiner = DiffusionPoseRefinerV20(
        j=J,
        in_dim=64,
        residual_hidden=16,
        num_diffusion_steps=8,
        num_inference_steps=2,
        beta_schedule="cosine",
    )
    out = refiner(pose, feat=feat)
    assert out.shape == (B, T, J, 3)
