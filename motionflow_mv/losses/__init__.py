"""Reusable loss functions for multi-view pose training."""

from .bone_length import bone_length_loss
from .canonical_skeleton_loss import canonical_skeleton_loss
from .focal_calibration_loss import focal_calibration_loss
from .gaussian_splatting_pose_loss import gaussian_splatting_pose_loss
from .masked_view_completion import masked_view_completion_loss
from .multiperson_association_loss import MultiPersonAssociationLoss
from .reprojection import reprojection_loss
from .temporal_consistency import (
    TemporalConsistencyLoss,
    acceleration_loss,
    temporal_consistency_loss,
)
from .velocity import velocity_loss, velocity_l1_loss
from .visibility_supervision_loss import visibility_supervision_loss

__all__ = [
    "TemporalConsistencyLoss",
    "acceleration_loss",
    "bone_length_loss",
    "canonical_skeleton_loss",
    "focal_calibration_loss",
    "gaussian_splatting_pose_loss",
    "masked_view_completion_loss",
    "reprojection_loss",
    "temporal_consistency_loss",
    "velocity_loss",
    "velocity_l1_loss",
    "visibility_supervision_loss",
]
