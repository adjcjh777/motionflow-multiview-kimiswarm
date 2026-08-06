from .voxelpose_loader import VoxelPoseShelfLoader
from .webbridge_loader import (
    convert_human36m,
    convert_shelf_campus,
    convert_synthetic_amass,
    convert_panoptic,
    convert_3dpw,
)

__all__ = [
    "VoxelPoseShelfLoader",
    "convert_human36m",
    "convert_shelf_campus",
    "convert_synthetic_amass",
    "convert_panoptic",
    "convert_3dpw",
]
