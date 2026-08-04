"""Lightweight camera model for multi-view calibration."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    """Pinhole camera defined by intrinsics K and extrinsics (R, t)."""

    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @property
    def projection_matrix(self) -> np.ndarray:
        """Return the 3x4 projection matrix P = K [R | t]."""
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt
