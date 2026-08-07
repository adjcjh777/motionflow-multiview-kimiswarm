"""Identity-initialization contract for learned camera corrections."""

import torch

from motionflow_mv.fusion.camera_centric_coordinate_transform import (
    CameraCentricCoordinateTransform,
)
from motionflow_mv.fusion.intrinsic_correction import IntrinsicCorrection


def test_intrinsic_correction_starts_at_identity():
    dtype = torch.float64
    K = torch.tensor(
        [[[[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]]],
        dtype=dtype,
    ).repeat(2, 3, 1, 1)
    x = torch.rand(2, 3, 5, 3, dtype=dtype)
    layer = IntrinsicCorrection(d=16, hidden=8).to(dtype=dtype)

    K_corrected, pp_delta, focal_scale = layer(K, x=x, weights=x[..., 2])

    torch.testing.assert_close(K_corrected, K, rtol=0, atol=0)
    torch.testing.assert_close(pp_delta, torch.zeros_like(pp_delta), rtol=0, atol=0)
    torch.testing.assert_close(focal_scale, torch.ones_like(focal_scale), rtol=0, atol=0)


def test_camera_centric_transform_starts_at_identity():
    dtype = torch.float64
    batch, views, joints = 2, 3, 5
    R = torch.eye(3, dtype=dtype).view(1, 1, 3, 3).repeat(batch, views, 1, 1)
    t = torch.randn(batch, views, 3, dtype=dtype)
    K = torch.tensor(
        [[[[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]]],
        dtype=dtype,
    ).repeat(batch, views, 1, 1)
    x = torch.rand(batch, views, joints, 3, dtype=dtype)
    layer = CameraCentricCoordinateTransform(
        d=16,
        hidden=8,
        condition_on_deep_features=False,
    ).to(dtype=dtype)

    R_corrected, t_corrected, scale, delta_R, delta_t, scale_factor = layer(
        R,
        t,
        x=x,
        K=K,
        weights=x[..., 2],
    )

    identity = torch.eye(3, dtype=dtype).view(1, 1, 3, 3).expand_as(delta_R)
    torch.testing.assert_close(R_corrected, R, rtol=0, atol=0)
    torch.testing.assert_close(t_corrected, t, rtol=0, atol=0)
    torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0, atol=0)
    torch.testing.assert_close(delta_R, identity, rtol=0, atol=0)
    torch.testing.assert_close(delta_t, torch.zeros_like(delta_t), rtol=0, atol=0)
    torch.testing.assert_close(scale_factor, torch.zeros_like(scale_factor), rtol=0, atol=0)
