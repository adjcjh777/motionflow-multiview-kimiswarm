"""Training-time calibration perturbation curriculum for v4.

Wraps the low-level camera perturbation routines with an epoch-based
curriculum so the v4 trainer can progressively increase rotation,
focal-length and principal-point noise.  The curriculum is checkpointable
and reports the current standard deviations for logging.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from motionflow_mv.calibration.camera_perturbation_curriculum import (
    extended_camera_perturbation_schedule,
    extended_camera_perturbation_schedule_with_anneal,
)
from motionflow_mv.calibration.perturb import perturb_cameras


class CalibrationPerturbationCurriculum:
    """Stateful curriculum for training-time camera perturbations.

    The curriculum increases ``rot_deg``, ``focal_pct`` and ``pp_px``
    perturbations over epochs, with optional warmup and cosine annealing.
    The v4 trainer can call :meth:`set_epoch` (or :meth:`step_epoch`) at
    the start of each epoch and then apply the perturbation with
    ``K_aug, R_aug, t_aug = curriculum(K, R, t)``.

    Args:
        rot_deg: Maximum rotation perturbation standard deviation in degrees.
        focal_pct: Maximum relative focal-length perturbation
            (e.g. ``0.05`` = 5%).
        pp_px: Maximum principal-point perturbation in pixels.
        trans: Maximum translation perturbation (same unit as camera
            translation).
        schedule: One of ``"flat"``, ``"extrinsic_curriculum"``,
            ``"intrinsics_curriculum"``, ``"extended_curriculum"`` or
            ``"extended_intrinsics_curriculum"``.
        ramp_epochs: Number of epochs over which extrinsics ramp up.
        intrinsics_ramp_epochs: Number of epochs over which intrinsics
            ramp up.
        focal_ramp_epochs: Optional separate ramp for focal length.
        pp_ramp_epochs: Optional separate ramp for principal point.
        warmup_epochs: Epochs with zero perturbation at the start.
        total_epochs: If provided, cosine annealing is applied after the ramp.
        epoch: Initial epoch.
    """

    def __init__(
        self,
        *,
        rot_deg: float = 0.5,
        focal_pct: float = 0.01,
        pp_px: float = 2.0,
        trans: float = 0.005,
        schedule: str = "extended_curriculum",
        ramp_epochs: int = 10,
        intrinsics_ramp_epochs: int = 5,
        focal_ramp_epochs: Optional[int] = None,
        pp_ramp_epochs: Optional[int] = None,
        warmup_epochs: int = 0,
        total_epochs: Optional[int] = None,
        epoch: int = 0,
    ) -> None:
        self.max_rot = rot_deg
        self.max_focal = focal_pct
        self.max_pp = pp_px
        self.max_trans = trans
        self.schedule = schedule
        self.ramp_epochs = ramp_epochs
        self.intrinsics_ramp_epochs = intrinsics_ramp_epochs
        self.focal_ramp_epochs = focal_ramp_epochs
        self.pp_ramp_epochs = pp_ramp_epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.epoch = epoch
        self._stds: Dict[str, float] = self._compute_stds(epoch)

    def _compute_stds(self, epoch: int) -> Dict[str, float]:
        if self.total_epochs is not None:
            return extended_camera_perturbation_schedule_with_anneal(
                epoch,
                total_epochs=self.total_epochs,
                schedule=self.schedule,
                rot=self.max_rot,
                trans=self.max_trans,
                focal=self.max_focal,
                pp=self.max_pp,
                ramp_epochs=self.ramp_epochs,
                intrinsics_ramp_epochs=self.intrinsics_ramp_epochs,
                focal_ramp_epochs=self.focal_ramp_epochs,
                pp_ramp_epochs=self.pp_ramp_epochs,
                warmup_epochs=self.warmup_epochs,
            )
        return extended_camera_perturbation_schedule(
            epoch,
            schedule=self.schedule,
            rot=self.max_rot,
            trans=self.max_trans,
            focal=self.max_focal,
            pp=self.max_pp,
            ramp_epochs=self.ramp_epochs,
            intrinsics_ramp_epochs=self.intrinsics_ramp_epochs,
            focal_ramp_epochs=self.focal_ramp_epochs,
            pp_ramp_epochs=self.pp_ramp_epochs,
            warmup_epochs=self.warmup_epochs,
            cosine_anneal=False,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the curriculum to a specific epoch."""
        self.epoch = epoch
        self._stds = self._compute_stds(epoch)

    def step_epoch(self) -> None:
        """Advance the curriculum by one epoch."""
        self.set_epoch(self.epoch + 1)

    @property
    def stds(self) -> Dict[str, float]:
        """Return a copy of the current standard-deviation dict."""
        return self._stds.copy()

    @property
    def rot_std(self) -> float:
        return self._stds["rot_std"]

    @property
    def trans_std(self) -> float:
        return self._stds["trans_std"]

    @property
    def focal_std(self) -> float:
        return self._stds["focal_std"]

    @property
    def pp_std(self) -> float:
        return self._stds["pp_std"]

    def __call__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        *,
        epoch: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the current curriculum perturbations to cameras.

        Args:
            K: Intrinsics ``(..., 3, 3)`` or ``(B, V, 3, 3)``.
            R: Rotations ``(..., 3, 3)`` or ``(B, V, 3, 3)``.
            t: Translations ``(..., 3)`` or ``(B, V, 3)``.
            epoch: Optionally override the stored epoch for this call.

        Returns:
            Perturbed ``(K, R, t)`` with the same shapes as the inputs.
        """
        if epoch is not None:
            self.set_epoch(epoch)
        stds = self._stds
        return perturb_cameras(
            K,
            R,
            t,
            rot_std=stds["rot_std"],
            trans_std=stds["trans_std"],
            focal_std=stds["focal_std"],
            pp_std=stds["pp_std"],
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return serialisable curriculum state."""
        return {
            "epoch": self.epoch,
            "max_rot": self.max_rot,
            "max_focal": self.max_focal,
            "max_pp": self.max_pp,
            "max_trans": self.max_trans,
            "schedule": self.schedule,
            "ramp_epochs": self.ramp_epochs,
            "intrinsics_ramp_epochs": self.intrinsics_ramp_epochs,
            "focal_ramp_epochs": self.focal_ramp_epochs,
            "pp_ramp_epochs": self.pp_ramp_epochs,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore curriculum state."""
        self.epoch = state_dict["epoch"]
        self.max_rot = state_dict["max_rot"]
        self.max_focal = state_dict["max_focal"]
        self.max_pp = state_dict["max_pp"]
        self.max_trans = state_dict["max_trans"]
        self.schedule = state_dict["schedule"]
        self.ramp_epochs = state_dict["ramp_epochs"]
        self.intrinsics_ramp_epochs = state_dict["intrinsics_ramp_epochs"]
        self.focal_ramp_epochs = state_dict.get("focal_ramp_epochs")
        self.pp_ramp_epochs = state_dict.get("pp_ramp_epochs")
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.total_epochs = state_dict["total_epochs"]
        self._stds = self._compute_stds(self.epoch)


if __name__ == "__main__":
    import torch

    K = torch.eye(3).unsqueeze(0).expand(4, 3, 3).contiguous()
    K[:, 0, 0] = 1000.0
    K[:, 1, 1] = 1000.0
    K[:, 0, 2] = 640.0
    K[:, 1, 2] = 360.0
    R = torch.eye(3).unsqueeze(0).expand(4, 3, 3).contiguous()
    t = torch.zeros(4, 3)

    curriculum = CalibrationPerturbationCurriculum(
        rot_deg=2.0,
        focal_pct=0.05,
        pp_px=10.0,
        schedule="extended_curriculum",
        ramp_epochs=5,
        intrinsics_ramp_epochs=3,
        warmup_epochs=2,
    )
    curriculum.set_epoch(10)
    K_aug, R_aug, t_aug = curriculum(K, R, t)
    print("stds:", curriculum.stds)
    print("K_aug shape:", K_aug.shape)
    print("R_aug shape:", R_aug.shape)
    print("t_aug shape:", t_aug.shape)
