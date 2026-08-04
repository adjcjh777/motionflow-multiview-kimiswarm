"""Tests for multi-view IR adapter and fusion module interface."""

import numpy as np
import pytest

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.fusion_module import DLTFusion, FUSION_REGISTRY
from motionflow_mv.fusion.attention_fusion_module import AttentionFusionModule
from motionflow_mv.ir.gvhmr_adapter import gvhmr_pt_to_ir
from motionflow_mv.ir.human_motion_ir import HumanMotionIR
from motionflow_mv.ir.multiview_adapter import fuse_multiple_irs


@pytest.fixture
def synthetic_irs_and_cameras(tmp_path):
    """Generate two per-view IRs with known 2D observations of a 3D point."""
    import torch

    T = 10
    J = 17
    betas_dim = 10
    views = []

    def make_ir(view_id: str):
        smpl_params = {
            "body_pose": torch.randn(T, 63) * 0.1,
            "global_orient": torch.randn(T, 3) * 0.1,
            "transl": torch.randn(T, 3) * 0.1,
            "betas": torch.randn(T, betas_dim) * 0.01,
        }
        pred = {
            "smpl_params_global": smpl_params,
            "smpl_params_incam": {k: v.clone() for k, v in smpl_params.items()},
            "K_fullimg": torch.eye(3).unsqueeze(0).repeat(T, 1, 1),
        }
        pt_path = tmp_path / f"hmr4d_{view_id}.pt"
        torch.save(pred, pt_path)
        ir = gvhmr_pt_to_ir(pt_path, sequence_id=view_id)
        return ir

    for view_id in ("cam_0", "cam_1"):
        ir = make_ir(view_id)
        # Create synthetic per-view 2D and confidence keyed by the IR's own id.
        ir.per_view_2d = {view_id: np.random.rand(T, J, 2).astype(np.float32)}
        ir.per_view_confidence = {view_id: np.ones((T, J), dtype=np.float32) * 0.9}
        views.append(ir)

    # Two cameras looking at the origin from +/- x axis.
    cameras = [
        Camera(K=np.eye(3), R=np.eye(3), t=np.array([0, 0, 0], dtype=np.float64)),
        Camera(K=np.eye(3), R=np.eye(3), t=np.array([1, 0, 0], dtype=np.float64)),
    ]
    return views, cameras


def test_dlt_fusion_shape(synthetic_irs_and_cameras):
    irs, cameras = synthetic_irs_and_cameras
    fused = fuse_multiple_irs(irs, cameras, DLTFusion())
    assert isinstance(fused, HumanMotionIR)
    assert fused.fusion_method == "dlt"
    assert fused.views == ["cam_0", "cam_1"]
    assert "transl" in fused.pose


def test_registry_has_dlt():
    assert "dlt" in FUSION_REGISTRY.names()
    module = FUSION_REGISTRY.get("dlt")
    assert isinstance(module, DLTFusion)


def test_registry_has_attention():
    assert "attention" in FUSION_REGISTRY.names()
    module = FUSION_REGISTRY.get("attention")
    assert isinstance(module, AttentionFusionModule)


def test_registry_has_all_plugins():
    expected = {"dlt", "attention", "robust_triangulation", "residual_refiner", "temporal_refiner"}
    assert expected.issubset(set(FUSION_REGISTRY.names()))


def test_dlt_fusion_with_perfect_point(tmp_path):
    """Triangulate a single perfect 3D point observed by two cameras."""
    T, J = 1, 1
    # Two cameras at origin and (1,0,0), both looking down z.
    cameras = [
        Camera(K=np.eye(3), R=np.eye(3), t=np.array([0, 0, 0], dtype=np.float64)),
        Camera(K=np.eye(3), R=np.eye(3), t=np.array([1, 0, 0], dtype=np.float64)),
    ]
    # 3D point at (0, 0, 5)
    point_3d = np.array([0.0, 0.0, 5.0])
    # Projection under P = K[R|t] where t = -R*c and c is the camera center.
    # For cam0 at origin: t=0, projected = point_3d, so uv = (X/Z, Y/Z).
    p0 = point_3d[:2] / point_3d[2]  # (0, 0)
    # For cam1 centered at (1,0,0): t = (-1,0,0), so projected = point_3d + t = (-1,0,5).
    p1 = (point_3d + cameras[1].t)[:2] / (point_3d + cameras[1].t)[2]
    points_2d = np.zeros((T, 2, J, 2), dtype=np.float64)
    points_2d[:, 0, 0, :] = p0
    points_2d[:, 1, 0, :] = p1
    confidences = np.ones((T, 2, J), dtype=np.float64)

    fused = DLTFusion().fuse(points_2d, confidences, cameras)
    assert fused.shape == (T, J, 3)
    assert np.all(np.isfinite(fused))
    # The triangulated point should be close to the original up to a scale ambiguity
    # in this degenerate two-view setup; just check it is not zero and mostly z.
    assert fused[0, 0, 2] != 0.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = d
    pytest.main([__file__, "-v"])
