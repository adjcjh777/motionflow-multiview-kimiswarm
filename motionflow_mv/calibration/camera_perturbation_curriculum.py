"""Extended camera perturbation ranges and curriculum schedules."""

from __future__ import annotations

import math
from typing import Any


def extended_camera_perturbation_schedule(
    epoch: int,
    /,
    schedule: str,
    *,
    rot: float = 0.5,
    trans: float = 0.005,
    focal: float = 0.01,
    pp: float = 2.0,
    ramp_epochs: int = 10,
    intrinsics_ramp_epochs: int = 5,
    focal_ramp_epochs: int | None = None,
    pp_ramp_epochs: int | None = None,
    warmup_epochs: int = 0,
    cosine_anneal: bool = False,
) -> dict[str, float]:
    """Return the per-epoch camera perturbation standard deviations."""
    if schedule == "flat":
        return {"rot_std": rot, "trans_std": trans, "focal_std": focal, "pp_std": pp}

    if schedule == "extrinsic_curriculum":
        ramp = min(1.0, max(0.0, epoch - warmup_epochs) / max(1, ramp_epochs))
        return {
            "rot_std": rot * ramp,
            "trans_std": trans * ramp,
            "focal_std": focal,
            "pp_std": pp,
        }

    if schedule == "intrinsics_curriculum":
        ramp = min(1.0, max(0.0, epoch - warmup_epochs) / max(1, intrinsics_ramp_epochs))
        return {
            "rot_std": rot,
            "trans_std": trans,
            "focal_std": focal * ramp,
            "pp_std": pp * ramp,
        }

    if schedule in ("extended_curriculum", "extended_intrinsics_curriculum"):
        focal_ramp = focal_ramp_epochs if focal_ramp_epochs is not None else intrinsics_ramp_epochs
        pp_ramp = pp_ramp_epochs if pp_ramp_epochs is not None else intrinsics_ramp_epochs

        if schedule == "extended_curriculum":
            ext_ramp = min(1.0, max(0.0, epoch - warmup_epochs) / max(1, ramp_epochs))
            rot_std = rot * ext_ramp
            trans_std = trans * ext_ramp
        else:
            rot_std = rot
            trans_std = trans

        int_ramp_epoch = max(0, epoch - warmup_epochs)
        focal_ramp_factor = min(1.0, int_ramp_epoch / max(1, focal_ramp))
        pp_ramp_factor = min(1.0, int_ramp_epoch / max(1, pp_ramp))

        result = {
            "rot_std": rot_std,
            "trans_std": trans_std,
            "focal_std": focal * focal_ramp_factor,
            "pp_std": pp * pp_ramp_factor,
        }

        if cosine_anneal:
            raise ValueError(
                "cosine_anneal=True requires total_epochs; use "
                "extended_camera_perturbation_schedule_with_anneal() instead."
            )

        return result

    raise ValueError(f"Unknown camera perturbation schedule: {schedule!r}")


def extended_camera_perturbation_schedule_with_anneal(
    epoch: int,
    /,
    *,
    total_epochs: int,
    schedule: str = "extended_curriculum",
    rot: float = 2.0,
    trans: float = 0.02,
    focal: float = 0.05,
    pp: float = 10.0,
    ramp_epochs: int = 10,
    intrinsics_ramp_epochs: int = 5,
    focal_ramp_epochs: int | None = None,
    pp_ramp_epochs: int | None = None,
    warmup_epochs: int = 2,
) -> dict[str, float]:
    """Same as extended_camera_perturbation_schedule but with cosine annealing."""
    base = extended_camera_perturbation_schedule(
        epoch,
        schedule=schedule,
        rot=rot,
        trans=trans,
        focal=focal,
        pp=pp,
        ramp_epochs=ramp_epochs,
        intrinsics_ramp_epochs=intrinsics_ramp_epochs,
        focal_ramp_epochs=focal_ramp_epochs,
        pp_ramp_epochs=pp_ramp_epochs,
        warmup_epochs=warmup_epochs,
        cosine_anneal=False,
    )

    anneal_start = warmup_epochs + max(ramp_epochs, intrinsics_ramp_epochs)
    if total_epochs <= anneal_start:
        return base

    t = max(0.0, min(1.0, (epoch - anneal_start) / max(1, total_epochs - anneal_start)))
    anneal = 0.7 + 0.3 * (0.5 * (1.0 + math.cos(math.pi * t)))
    return {k: v * anneal for k, v in base.items()}


def schedule_from_args(
    epoch: int,
    /,
    args: Any,
    *,
    total_epochs: int | None = None,
) -> dict[str, float]:
    """Build a schedule from an argparse namespace."""
    schedule = getattr(args, "cam_aug_schedule", "flat")
    extended_schedules = ("extended_curriculum", "extended_intrinsics_curriculum")

    if schedule not in extended_schedules:
        raise ValueError(
            f"schedule_from_args only handles extended schedules; got {schedule!r}"
        )

    kwargs = dict(
        schedule=schedule,
        rot=getattr(args, "cam_aug_rot", 0.5),
        trans=getattr(args, "cam_aug_trans", 0.005),
        focal=getattr(args, "cam_aug_focal", 0.01),
        pp=getattr(args, "cam_aug_pp", 2.0),
        ramp_epochs=getattr(args, "cam_aug_ramp_epochs", 10),
        intrinsics_ramp_epochs=getattr(args, "cam_aug_intrinsics_ramp_epochs", 5),
        focal_ramp_epochs=getattr(args, "cam_aug_focal_ramp_epochs", None),
        pp_ramp_epochs=getattr(args, "cam_aug_pp_ramp_epochs", None),
        warmup_epochs=getattr(args, "cam_aug_warmup_epochs", 0),
    )

    if total_epochs is not None:
        return extended_camera_perturbation_schedule_with_anneal(
            epoch, total_epochs=total_epochs, **kwargs
        )
    return extended_camera_perturbation_schedule(epoch, **kwargs)
