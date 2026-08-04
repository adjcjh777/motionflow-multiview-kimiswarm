"""HumanMotionIR: a stable intermediate representation for human motion.

This IR decouples upstream human recovery (GVHMR, ScoreHMR, etc.) from
downstream robot retargeting and policy training. It is the integration
contract between single-view/multi-view fusion modules and the rest of the
MotionFlow pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class HumanMotionIR:
    """Stable intermediate representation for a single human motion sequence.

    Attributes:
        schema_version: version of the IR schema.
        sequence_id: unique identifier for the sequence.
        person_id: unique identifier for the person (for multi-person scenes).
        fps: frame rate in Hz.
        timestamps: (T,) array of timestamps in seconds.
        human_model: 'smpl', 'smplx', or other.
        pose: dict of pose parameters. For SMPL/SMPL-X:
            - body_pose: (T, J*3) or (T, J, 3)
            - global_orient: (T, 3) or (T, 1, 3)
            - transl: (T, 3)
            - betas: (T, B) or (B,)
        coordinate_system: dict describing the world coordinate system.
            - up_axis, forward_axis, length_unit, world_from_reference (4x4)
        views: list of view identifiers (e.g. camera names) used for multi-view fusion.
        camera_parameters: per-view camera parameters (optional).
        per_view_2d: per-view 2D keypoint observations (optional), keyed by view id.
        per_view_confidence: per-view 2D confidence scores (optional), keyed by view id.
        fusion_method: name of the fusion method used to produce this IR.
        uncertainty: dict of optional quality/uncertainty fields.
        quality: dict of quality flags and summary metrics.
        provenance: dict of version/lineage metadata.
    """

    schema_version: str
    sequence_id: str
    person_id: str
    fps: float
    timestamps: np.ndarray
    human_model: str
    pose: Dict[str, np.ndarray]
    coordinate_system: Dict[str, Any]
    views: List[str] = field(default_factory=list)
    camera_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_view_2d: Optional[Dict[str, np.ndarray]] = None
    per_view_confidence: Optional[Dict[str, np.ndarray]] = None
    fusion_method: str = "single_view"
    uncertainty: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.fps > 0
        assert len(self.timestamps) > 0
