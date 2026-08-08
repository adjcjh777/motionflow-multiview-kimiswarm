import pytest
import torch

from motionflow_mv.fusion.ray_conditioned_attention_v33 import RayConditionedCrossViewAttentionV33


def test_ray_conditioned_attention_identity_shape():
    b, t, v, j, d = 2, 3, 4, 17, 64
    module = RayConditionedCrossViewAttentionV33(d=d, n_heads=4, n_layers=2)
    tokens = torch.randn(b, t, v, j, d)
    points_2d = torch.randn(b, t, v, j, 2)
    K = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, v, 3, 3).clone()
    K[:, :, :, 0, 0] = 800.0
    K[:, :, :, 1, 1] = 800.0
    K[:, :, :, 0, 2] = 320.0
    K[:, :, :, 1, 2] = 240.0
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, v, 3, 3).clone()
    t_vec = torch.zeros(b, t, v, 3)

    out = module(tokens, points_2d, K, R, t_vec, view_mask=None)
    assert out.shape == (b, t, v, j, d)
    # Identity-at-init: gate init -6 -> sigmoid ~0.002, so the residual is tiny.
    assert torch.allclose(out, tokens, atol=0.1)


def test_ray_conditioned_attention_with_view_mask():
    b, t, v, j, d = 2, 1, 4, 17, 64
    module = RayConditionedCrossViewAttentionV33(d=d, n_heads=4, n_layers=1)
    tokens = torch.randn(b, t, v, j, d)
    points_2d = torch.randn(b, t, v, j, 2)
    K = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, v, 3, 3).clone()
    K[:, :, :, 0, 0] = 800.0
    K[:, :, :, 1, 1] = 800.0
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, v, 3, 3).clone()
    t_vec = torch.zeros(b, t, v, 3)
    view_mask = torch.tensor([[[1, 1, 1, 0], [1, 1, 0, 0]]], dtype=torch.float32)

    out = module(tokens, points_2d, K, R, t_vec, view_mask=view_mask)
    assert out.shape == (b, t, v, j, d)
