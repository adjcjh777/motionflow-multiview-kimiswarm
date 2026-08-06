"""CPU smoke tests for the extended camera perturbation curriculum."""

import pytest
import torch

from motionflow_mv.calibration.camera_perturbation_curriculum import (
    extended_camera_perturbation_schedule,
    extended_camera_perturbation_schedule_with_anneal,
    schedule_from_args,
)
from motionflow_mv.calibration.perturb import perturb_cameras_with_delta


class Args:
    cam_aug_schedule = "extended_curriculum"
    cam_aug_rot = 2.0
    cam_aug_trans = 0.02
    cam_aug_focal = 0.05
    cam_aug_pp = 10.0
    cam_aug_ramp_epochs = 5
    cam_aug_intrinsics_ramp_epochs = 3
    cam_aug_warmup_epochs = 2
    cam_aug_focal_ramp_epochs = None
    cam_aug_pp_ramp_epochs = None


def test_flat_schedule():
    s = extended_camera_perturbation_schedule(5, schedule="flat", rot=1.0, trans=0.01, focal=0.02, pp=5.0)
    assert s == {"rot_std": 1.0, "trans_std": 0.01, "focal_std": 0.02, "pp_std": 5.0}


def test_extended_curriculum_ramps_and_warmup():
    s1 = extended_camera_perturbation_schedule(
        1, schedule="extended_curriculum", rot=2.0, trans=0.02, focal=0.05, pp=10.0,
        ramp_epochs=5, intrinsics_ramp_epochs=3, warmup_epochs=2,
    )
    assert s1["rot_std"] == 0.0
    assert s1["focal_std"] == 0.0

    s2 = extended_camera_perturbation_schedule(
        7, schedule="extended_curriculum", rot=2.0, trans=0.02, focal=0.05, pp=10.0,
        ramp_epochs=5, intrinsics_ramp_epochs=3, warmup_epochs=2,
    )
    assert s2["rot_std"] == pytest.approx(2.0)
    assert s2["trans_std"] == pytest.approx(0.02)
    assert s2["focal_std"] == pytest.approx(0.05)
    assert s2["pp_std"] == pytest.approx(10.0)

    s3 = extended_camera_perturbation_schedule(
        4, schedule="extended_curriculum", rot=2.0, focal=0.05, pp=10.0,
        ramp_epochs=5, intrinsics_ramp_epochs=3, warmup_epochs=2,
    )
    assert 0.0 < s3["rot_std"] < 2.0
    assert 0.0 < s3["focal_std"] <= 0.05


def test_extended_intrinsics_curriculum_keeps_extrinsics_flat():
    s = extended_camera_perturbation_schedule(
        1, schedule="extended_intrinsics_curriculum", rot=1.5, trans=0.01, focal=0.05, pp=10.0,
    )
    assert s["rot_std"] == 1.5
    assert s["trans_std"] == 0.01
    assert 0.0 < s["focal_std"] <= 0.05
    assert 0.0 < s["pp_std"] <= 10.0


def test_annealed_schedule_decays():
    s_start = extended_camera_perturbation_schedule_with_anneal(
        10, total_epochs=20, schedule="extended_curriculum", rot=2.0, ramp_epochs=5, warmup_epochs=2
    )
    s_end = extended_camera_perturbation_schedule_with_anneal(
        20, total_epochs=20, schedule="extended_curriculum", rot=2.0, ramp_epochs=5, warmup_epochs=2
    )
    assert s_end["rot_std"] < s_start["rot_std"]
    assert s_end["rot_std"] > 0.0


def test_schedule_from_args_matches_manual():
    args = Args()
    manual = extended_camera_perturbation_schedule(
        10, schedule="extended_curriculum",
        rot=args.cam_aug_rot, trans=args.cam_aug_trans, focal=args.cam_aug_focal, pp=args.cam_aug_pp,
        ramp_epochs=args.cam_aug_ramp_epochs, intrinsics_ramp_epochs=args.cam_aug_intrinsics_ramp_epochs,
        warmup_epochs=args.cam_aug_warmup_epochs,
    )
    from_args = schedule_from_args(10, args)
    assert from_args == manual


def test_extended_perturbation_changes_cameras():
    K = torch.eye(3).unsqueeze(0).expand(4, 3, 3).contiguous()
    K[:, 0, 0] = 1000.0
    K[:, 1, 1] = 1000.0
    K[:, 0, 2] = 640.0
    K[:, 1, 2] = 360.0
    R = torch.eye(3).unsqueeze(0).expand(4, 3, 3).contiguous()
    t = torch.zeros(4, 3)
    schedule = extended_camera_perturbation_schedule(
        10, schedule="extended_curriculum", rot=2.0, trans=0.02, focal=0.05, pp=10.0,
        ramp_epochs=5, intrinsics_ramp_epochs=3, warmup_epochs=2,
    )
    K_aug, R_aug, t_aug, pp_delta, focal_scale = perturb_cameras_with_delta(
        K, R, t,
        rot_std=schedule["rot_std"],
        trans_std=schedule["trans_std"],
        focal_std=schedule["focal_std"],
        pp_std=schedule["pp_std"],
    )
    assert K_aug.shape == K.shape
    assert R_aug.shape == R.shape
    assert t_aug.shape == t.shape
    assert torch.any(K_aug != K) or torch.any(t_aug != t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
