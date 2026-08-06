"""CPU sanity tests for the camera-parameter-conditioned anchor model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_camera_conditioned_model import (
    RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_camera_conditioned_module import (
    RayAttentionTemporalCrossviewResidualCameraConditionedFusionModule,
)


def make_data(batch=1, t=3, v=2, j=17):
    x = torch.randn(batch, t, v, j, 3)
    x[..., 2] = (x[..., 2] + 1).clamp(min=0)  # confidences >= 0
    y = torch.randn(batch, t, j, 3)
    K = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    R = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    t_vec = torch.zeros(v, 3).float()
    return x, y, K, R, t_vec


def test_camera_conditioned_model_forward_backward():
    """One forward/backward step using explicit K, R, t tensors."""
    x, y, K, R, t = make_data(batch=2, t=3, v=2, j=17)
    model = RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned(
        j=17, d=32, n_views=2, n_st_layers=1, residual_hidden=64, return_pp_delta=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred, weights, pp_delta = model(x, K=K, R=R, t=t)
    assert pred.shape == (2, 3, 17, 3)
    assert weights.shape == (2, 3, 2, 17)
    assert pp_delta.shape[0] == 2 * 3
    loss = (pred - y).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_camera_conditioned_model_cameras_argument():
    """Forward pass using the cameras= list API."""
    from motionflow_mv.calibration.camera import Camera

    x, _, K, R, t = make_data(batch=1, t=2, v=2, j=17)
    cameras = [
        Camera(K=K[0].numpy(), R=R[0].numpy(), t=t[0].numpy())
        for _ in range(2)
    ]
    model = RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned(
        j=17, d=32, n_views=2, n_st_layers=1, residual_hidden=64
    )
    pred, _ = model(x, cameras=cameras)
    assert pred.shape == (1, 2, 17, 3)


def test_camera_conditioned_fusion_module():
    """FusionModule wrapper round-trip."""
    import numpy as np
    from motionflow_mv.calibration.camera import Camera

    j, v, t = 17, 2, 5
    points_2d = np.random.randn(t, v, j, 2).astype(np.float32)
    confidences = np.abs(np.random.randn(t, v, j)).astype(np.float32)
    cameras = [
        Camera(K=np.eye(3, dtype=np.float32), R=np.eye(3, dtype=np.float32), t=np.zeros(3, dtype=np.float32))
        for _ in range(v)
    ]
    module = RayAttentionTemporalCrossviewResidualCameraConditionedFusionModule(
        j=j, d=32, n_views=v, n_st_layers=1, residual_hidden=64
    )
    pred = module.fuse(points_2d, confidences, cameras)
    assert pred.shape == (t, j, 3)


def main():
    test_camera_conditioned_model_forward_backward()
    print("test_camera_conditioned_model_forward_backward passed")
    test_camera_conditioned_model_cameras_argument()
    print("test_camera_conditioned_model_cameras_argument passed")
    test_camera_conditioned_fusion_module()
    print("test_camera_conditioned_fusion_module passed")
    print("All camera_conditioned_pp tests passed.")


if __name__ == "__main__":
    main()
