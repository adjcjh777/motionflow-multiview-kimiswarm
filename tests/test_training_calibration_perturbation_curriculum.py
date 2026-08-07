"""CPU unit tests for the v4 calibration perturbation curriculum.

Verifies that perturbation magnitudes grow with the epoch and that the
perturbed cameras still produce finite triangulations.
"""

import pytest
import torch

from motionflow_mv.fusion.triangulation import triangulate_dlt_torch
from motionflow_mv.training import CalibrationPerturbationCurriculum


def _make_cams(v: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a simple calibrated rig with ``v`` views."""
    K = torch.eye(3).unsqueeze(0).expand(v, 3, 3).contiguous()
    K[:, 0, 0] = 1000.0
    K[:, 1, 1] = 1000.0
    K[:, 0, 2] = 640.0
    K[:, 1, 2] = 360.0
    R = torch.eye(3).unsqueeze(0).expand(v, 3, 3).contiguous()
    t = torch.zeros(v, 3)
    # Spread cameras around the subject on the x-z plane.
    for i in range(v):
        angle = 2.0 * 3.14159265 * i / max(v, 1)
        t[i] = torch.tensor([torch.sin(torch.tensor(angle)) * 5.0,
                              0.0,
                              torch.cos(torch.tensor(angle)) * 5.0])
    return K, R, t


def _project_point(
    X: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project a 3D point through ``v`` cameras.

    Args:
        X: (3,) world point.
        K, R, t: (V, 3, 3) and (V, 3).

    Returns:
        (V, 2) image points.
    """
    V = K.shape[0]
    Xh = torch.cat([X, torch.ones(1)], dim=0)  # (4,)
    pts = []
    for v in range(V):
        P = K[v] @ torch.cat([R[v], t[v].unsqueeze(-1)], dim=-1)  # (3, 4)
        x = P @ Xh  # (3,)
        x = x[:2] / x[2]
        pts.append(x)
    return torch.stack(pts, dim=0)


def test_curriculum_stds_grow_with_epoch():
    curriculum = CalibrationPerturbationCurriculum(
        rot_deg=2.0,
        focal_pct=0.05,
        pp_px=10.0,
        schedule="extended_curriculum",
        ramp_epochs=5,
        intrinsics_ramp_epochs=3,
        warmup_epochs=2,
    )
    stds_warmup = curriculum.stds
    curriculum.set_epoch(1)
    assert curriculum.stds["rot_std"] == 0.0
    assert curriculum.stds["focal_std"] == 0.0

    curriculum.set_epoch(7)
    full_stds = curriculum.stds
    assert full_stds["rot_std"] == pytest.approx(2.0)
    assert full_stds["trans_std"] == pytest.approx(0.005)
    assert full_stds["focal_std"] == pytest.approx(0.05)
    assert full_stds["pp_std"] == pytest.approx(10.0)

    # Monotonic growth over the ramp.
    rots = [curriculum._compute_stds(e)["rot_std"] for e in range(8)]
    assert rots == sorted(rots)


def test_curriculum_applies_perturbations():
    K, R, t = _make_cams(4)
    curriculum = CalibrationPerturbationCurriculum(
        rot_deg=2.0,
        focal_pct=0.05,
        pp_px=10.0,
        schedule="extended_curriculum",
        ramp_epochs=5,
        intrinsics_ramp_epochs=3,
        warmup_epochs=2,
    )
    curriculum.set_epoch(7)
    K_aug, R_aug, t_aug = curriculum(K, R, t)

    assert K_aug.shape == K.shape
    assert R_aug.shape == R.shape
    assert t_aug.shape == t.shape
    assert torch.any(K_aug != K) or torch.any(t_aug != t)
    assert torch.all(torch.isfinite(K_aug))
    assert torch.all(torch.isfinite(R_aug))
    assert torch.all(torch.isfinite(t_aug))


def test_curriculum_triangulation_stays_finite():
    K, R, t = _make_cams(4)
    # Use a single world point roughly in front of the rig.
    X_true = torch.tensor([0.3, 1.0, 3.0], dtype=torch.float32)
    pts_2d = _project_point(X_true, K, R, t)

    curriculum = CalibrationPerturbationCurriculum(
        rot_deg=2.0,
        focal_pct=0.05,
        pp_px=10.0,
        trans=0.02,
        schedule="extended_curriculum",
        ramp_epochs=5,
        intrinsics_ramp_epochs=3,
        warmup_epochs=2,
    )
    curriculum.set_epoch(7)
    K_aug, R_aug, t_aug = curriculum(K, R, t)

    V = K_aug.shape[0]
    proj = torch.zeros(V, 3, 4)
    for v in range(V):
        P = K_aug[v] @ torch.cat([R_aug[v], t_aug[v].unsqueeze(-1)], dim=-1)
        proj[v] = P

    X_est = triangulate_dlt_torch(pts_2d, proj)
    assert torch.all(torch.isfinite(X_est))
    # Perturbations are moderate; triangulated point should stay roughly
    # near the true point.
    assert (X_est - X_true).norm() < 2.0


def test_curriculum_flat_schedule():
    curriculum = CalibrationPerturbationCurriculum(
        rot_deg=1.0,
        focal_pct=0.02,
        pp_px=5.0,
        schedule="flat",
    )
    curriculum.set_epoch(100)
    assert curriculum.stds == {
        "rot_std": 1.0,
        "trans_std": 0.005,
        "focal_std": 0.02,
        "pp_std": 5.0,
    }


def test_curriculum_state_dict_roundtrip():
    curriculum = CalibrationPerturbationCurriculum(
        rot_deg=2.0,
        focal_pct=0.05,
        pp_px=10.0,
        schedule="extended_curriculum",
        ramp_epochs=5,
        intrinsics_ramp_epochs=3,
        warmup_epochs=2,
        total_epochs=20,
    )
    curriculum.set_epoch(12)
    state = curriculum.state_dict()
    restored = CalibrationPerturbationCurriculum()
    restored.load_state_dict(state)
    assert restored.epoch == curriculum.epoch
    assert restored.stds == curriculum.stds


def test_curriculum_step_epoch():
    curriculum = CalibrationPerturbationCurriculum(
        rot_deg=2.0,
        schedule="extended_curriculum",
        ramp_epochs=5,
        warmup_epochs=0,
    )
    s0 = curriculum.rot_std
    curriculum.step_epoch()
    s1 = curriculum.rot_std
    assert s1 > s0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
