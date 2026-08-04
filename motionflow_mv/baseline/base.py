"""Abstract interface for the monocular pose estimator used by MotionFlow."""

from abc import ABC, abstractmethod
import numpy as np


class BasePoseEstimator(ABC):
    """Adapter for any monocular video / image pose estimator.

    Subclasses should wrap concrete implementations such as the internal
    MotionFlow pipeline, WHAM, VIBE or 4D-Humans.
    """

    @abstractmethod
    def extract(self, video_path: str) -> dict:
        """Extract per-frame 2D keypoints and confidence from a video.

        Args:
            video_path: path to the input video file.

        Returns:
            dict with keys:
                - "keypoints_2d": np.ndarray of shape (T, J, 2)
                - "confidence":   np.ndarray of shape (T, J)
                where T is the number of frames and J the number of joints.
        """
        pass
