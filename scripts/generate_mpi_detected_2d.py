#!/usr/bin/env python3
"""Regenerate MPI-INF-3DHP canonical .npz files with detected 2D keypoints.

The canonical MPI-INF-3DHP .npz files currently store the ground-truth ``annot2``
2D keypoints (projected from ``univ_annot3``).  For the standard *detected-2D*
protocol, these GT 2D points must be replaced by actual 2D detections from an
off-the-shelf pose detector (HRNet / OpenPose / MediaPipe / RTMPose) run on each
raw camera frame.

This script now supports a plug-in detector interface.  It will attempt, in
order, to use:

1. MediaPipe Pose (``mediapipe`` Python package).
2. OpenPose via ``cv2.dnn`` if the model files are present locally.
3. A graceful fallback that adds configurable pixel noise to the GT 2D points and
   prints a warning.

The output .npz structure is identical to the canonical WebBridge format so
that existing configs and loaders require no changes.

Usage
-----
    # Auto-select a detector; fall back to GT + noise if none is available.
    python scripts/generate_mpi_detected_2d.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
        --detector auto \
        --image_dir data/webbridge/mpi_inf_3dhp/raw

    # Explicitly use the fallback (GT + noise).
    python scripts/generate_mpi_detected_2d.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
        --detector fallback \
        --fallback_noise 2.0

    # Explicitly request MediaPipe.
    python scripts/generate_mpi_detected_2d.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
        --detector mediapipe \
        --image_dir data/webbridge/mpi_inf_3dhp/raw

TODO
----
* Add HRNet / RTMPose wrappers via mmpose.
* Improve the MediaPipe -> MPI-INF-3DHP mapping for spine / clavicle joints.
* Batch images across views for faster detector inference.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Joint mapping: MediaPipe Pose -> MPI-INF-3DHP 28 joints
# ---------------------------------------------------------------------------
# MediaPipe Pose has 33 landmarks.  The mapping below only covers joints where
# there is a clear anatomical match; unmapped MPI joints keep the GT 2D value
# (with a warning) so the output skeleton stays complete and canonical.
# MPI-INF-3DHP skeleton (WebBridge convention, 0-indexed):
#   0 spine3, 1 spine4, 2 spine2, 3 spine, 4 pelvis, 5 neck, 6 head,
#   7 head_top, 8 left_clavicle, 9 left_shoulder, 10 left_elbow,
#   11 left_wrist, 12 left_hand, 13 right_clavicle, 14 right_shoulder,
#   15 right_elbow, 16 right_wrist, 17 right_hand, 18 left_hip,
#   19 left_knee, 20 left_ankle, 21 left_foot, 22 left_toe, 23 right_hip,
#   24 right_knee, 25 right_ankle, 26 right_foot, 27 right_toe
MEDIAPIPE_TO_MPI = {
    6: 9,   # left_shoulder  -> left_shoulder
    7: 14,  # right_shoulder -> right_shoulder
    8: 10,  # left_elbow     -> left_elbow
    9: 15,  # right_elbow    -> right_elbow
    10: 11, # left_wrist     -> left_wrist
    11: 16, # right_wrist    -> right_wrist
    12: 12, # left_pinky     -> left_hand
    13: 17, # right_pinky    -> right_hand
    14: 12, # left_index     -> left_hand
    15: 17, # right_index    -> right_hand
    16: 12, # left_thumb     -> left_hand
    17: 17, # right_thumb    -> right_hand
    18: 18, # left_hip       -> left_hip
    19: 23, # right_hip      -> right_hip
    20: 19, # left_knee      -> left_knee
    21: 24, # right_knee     -> right_knee
    22: 20, # left_ankle     -> left_ankle
    23: 25, # right_ankle    -> right_ankle
    24: 21, # left_heel      -> left_foot
    25: 26, # right_heel     -> right_foot
    26: 22, # left_foot_index-> left_toe
    27: 27, # right_foot_index->right_toe
    0: 6,   # nose           -> head (approximation)
    1: 7,   # left_eye_inner -> head_top (approximation)
    2: 7,   # left_eye       -> head_top (approximation)
    3: 7,   # left_eye_outer -> head_top (approximation)
    4: 7,   # right_eye_inner-> head_top (approximation)
    5: 7,   # right_eye      -> head_top (approximation)
    32: 7,  # right_eye_outer-> head_top (approximation)
}


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------
class Detector(ABC):
    """Abstract 2D detector interface.

    Subclasses must implement ``__call__(image_paths, points_2d_gt)`` and
    return ``(points_2d, confidences)`` with shapes ``(V, J, 2)`` and ``(V, J)``.
    """

    def __init__(self, name: str, device: str = "cpu") -> None:
        self.name = name
        self.device = device

    @abstractmethod
    def __call__(
        self,
        image_paths: Sequence[Optional[Path]],
        points_2d_gt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run detection.

        Parameters
        ----------
        image_paths:
            Per-view image paths.  Missing images are represented by ``None``.
        points_2d_gt:
            Ground-truth 2D points for this frame, shape ``(V, J, 2)``.  Can be
            used to fill joints the detector does not model.

        Returns
        -------
        points_2d:
            Detected 2D keypoints, shape ``(V, J, 2)``.
        confidences:
            Per-joint detection confidences, shape ``(V, J)``.
        """
        ...


class FallbackDetector(Detector):
    """Fallback detector: returns GT 2D + configurable Gaussian pixel noise."""

    def __init__(self, noise_std: float = 2.0, device: str = "cpu") -> None:
        super().__init__("fallback", device=device)
        self.noise_std = noise_std

    def __call__(
        self,
        image_paths: Sequence[Optional[Path]],
        points_2d_gt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.noise_std <= 0:
            return points_2d_gt.copy(), np.ones(points_2d_gt.shape[:2], dtype=np.float32)
        noise = np.random.normal(
            scale=self.noise_std, size=points_2d_gt.shape
        ).astype(np.float64)
        conf = np.ones(points_2d_gt.shape[:2], dtype=np.float32) * 0.9
        return points_2d_gt.copy() + noise, conf


class MediaPipePoseDetector(Detector):
    """MediaPipe Pose 2D detector wrapper.

    Requires ``mediapipe`` to be installed.  The detector runs independently on
    each view image and maps the 33 MediaPipe landmarks to the MPI-INF-3DHP
    28-joint skeleton using ``MEDIAPIPE_TO_MPI``.  Joints that have no direct
    MediaPipe correspondent are filled from ``points_2d_gt`` so the output
    remains a complete canonical frame.
    """

    def __init__(
        self,
        device: str = "cpu",
        static_image_mode: bool = True,
        model_complexity: int = 1,
    ) -> None:
        super().__init__("mediapipe", device=device)
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Install it with: "
                "pip install mediapipe"
            ) from exc

        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _load_image(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None or not Path(path).exists():
            return None
        try:
            # Lazy import so that the script can still be parsed when cv2 is absent.
            import cv2  # type: ignore
            img = cv2.imread(str(path))
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def __call__(
        self,
        image_paths: Sequence[Optional[Path]],
        points_2d_gt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        V, J, _ = points_2d_gt.shape
        out = np.zeros((V, J, 2), dtype=np.float64)
        conf = np.zeros((V, J), dtype=np.float32)

        for v, path in enumerate(image_paths):
            img = self._load_image(path)
            if img is None:
                # Missing image: fall back to GT for this view.
                out[v] = points_2d_gt[v]
                conf[v] = 0.5
                continue

            results = self.pose.process(img)
            if results.pose_landmarks is None:
                # No detection: fall back to GT for this view.
                out[v] = points_2d_gt[v]
                conf[v] = 0.5
                continue

            landmarks = results.pose_landmarks.landmark
            h, w = img.shape[:2]
            mapped: set[int] = set()
            for mp_idx, mpi_idx in MEDIAPIPE_TO_MPI.items():
                if mpi_idx in mapped:
                    # Average with previous assignment for joints that receive
                    # multiple MediaPipe landmarks.
                    existing = out[v, mpi_idx]
                    lmk = landmarks[mp_idx]
                    new_pt = np.array([lmk.x * w, lmk.y * h], dtype=np.float64)
                    out[v, mpi_idx] = (existing + new_pt) / 2.0
                    conf[v, mpi_idx] = max(conf[v, mpi_idx], lmk.visibility)
                else:
                    lmk = landmarks[mp_idx]
                    out[v, mpi_idx] = np.array([lmk.x * w, lmk.y * h], dtype=np.float64)
                    conf[v, mpi_idx] = lmk.visibility
                mapped.add(mpi_idx)

            # Fill any unmapped MPI joints with GT so the skeleton is complete.
            for j in range(J):
                if j not in mapped:
                    out[v, j] = points_2d_gt[v, j]
                    conf[v, j] = 0.25

        return out, conf


class OpenPoseCVDNNDetector(Detector):
    """OpenPose detector via ``cv2.dnn``.

    Looks for the COCO model files (``pose/prototxt`` and ``pose/caffemodel``)
    under ``openpose_model_dir``.  If they are missing it raises an error so
    the auto-selector can fall back to the next available detector.
    """

    DEFAULT_MODEL_DIR = Path("models/openpose")

    def __init__(
        self,
        device: str = "cpu",
        model_dir: Optional[Path] = None,
        prototxt: Optional[str] = None,
        caffemodel: Optional[str] = None,
    ) -> None:
        super().__init__("openpose", device=device)
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install it with: pip install opencv-python"
            ) from exc

        model_dir = Path(model_dir or self.DEFAULT_MODEL_DIR)
        prototxt = prototxt or "openpose_coco.prototxt"
        caffemodel = caffemodel or "pose_iter_440000.caffemodel"
        proto_path = model_dir / prototxt
        model_path = model_dir / caffemodel

        if not proto_path.exists() or not model_path.exists():
            raise RuntimeError(
                f"OpenPose COCO model files not found in {model_dir}. "
                "Download them from https://github.com/CMU-Perceptual-Computing-Lab/openpose "
                "and place the .prototxt and .caffemodel in models/openpose/"
            )

        self.net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
        if device == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def _load_image(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None or not Path(path).exists():
            return None
        try:
            import cv2  # type: ignore
            return cv2.imread(str(path))
        except Exception:
            return None

    def __call__(
        self,
        image_paths: Sequence[Optional[Path]],
        points_2d_gt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # OpenPose COCO produces 18 joints.  We map them to MPI where possible.
        # COCO order: nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip,
        # Rknee, Rankle, Lhip, Lknee, Lankle, Reye, Leye, Rear, Lear.
        COCO_TO_MPI = {
            0: 6,   # nose      -> head
            1: 5,   # neck      -> neck
            2: 14,  # Rsho      -> right_shoulder
            3: 15,  # Relb      -> right_elbow
            4: 16,  # Rwri      -> right_wrist
            5: 9,   # Lsho      -> left_shoulder
            6: 10,  # Lelb      -> left_elbow
            7: 11,  # Lwri      -> left_wrist
            8: 23,  # Rhip      -> right_hip
            9: 24,  # Rknee     -> right_knee
            10: 25, # Rankle    -> right_ankle
            11: 18, # Lhip      -> left_hip
            12: 19, # Lknee     -> left_knee
            13: 20, # Lankle    -> left_ankle
        }
        V, J, _ = points_2d_gt.shape
        out = np.zeros((V, J, 2), dtype=np.float64)
        conf = np.zeros((V, J), dtype=np.float32)

        for v, path in enumerate(image_paths):
            img = self._load_image(path)
            if img is None:
                out[v] = points_2d_gt[v]
                conf[v] = 0.5
                continue

            h, w = img.shape[:2]
            inp = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (368, 368), (0, 0, 0), swapRB=False, crop=False)
            self.net.setInput(inp)
            output = self.net.forward()
            H, W = output.shape[2], output.shape[3]

            mapped: set[int] = set()
            for coco_idx, mpi_idx in COCO_TO_MPI.items():
                heatmap = output[0, coco_idx, :, :]
                _, conf_val, _, point = cv2.minMaxLoc(heatmap)
                x = point[0] * w / W
                y = point[1] * h / H
                out[v, mpi_idx] = np.array([x, y], dtype=np.float64)
                conf[v, mpi_idx] = float(conf_val)
                mapped.add(mpi_idx)

            for j in range(J):
                if j not in mapped:
                    out[v, j] = points_2d_gt[v, j]
                    conf[v, j] = 0.25

        return out, conf


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------
def _build_detector(name: str, device: str = "cpu", fallback_noise: float = 2.0) -> Detector:
    """Build a 2D detector by name.

    ``auto`` tries real detectors in order and falls back to GT+noise.
    """
    name = name.lower().strip()

    if name == "mediapipe":
        return MediaPipePoseDetector(device=device)
    if name == "openpose":
        return OpenPoseCVDNNDetector(device=device)
    if name in {"stub", "none", "", "fallback"}:
        return FallbackDetector(noise_std=fallback_noise, device=device)

    if name == "auto":
        errors: List[str] = []
        for ctor, label in (
            (MediaPipePoseDetector, "MediaPipe Pose"),
            (OpenPoseCVDNNDetector, "OpenPose cv2.dnn"),
        ):
            try:
                return ctor(device=device)
            except Exception as exc:  # pragma: no cover - detector may be absent
                errors.append(f"{label}: {exc}")
        print(
            "WARNING: No real 2D detector is available. "
            f"Falling back to GT 2D + {fallback_noise}px noise.\n"
            "Details:\n  " + "\n  ".join(errors)
        )
        return FallbackDetector(noise_std=fallback_noise, device=device)

    raise ValueError(f"Unknown detector: {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_OMIT_RE = re.compile(r"_smoke|_v4_|test_set")


def _find_source_npz(input_dir: Path) -> List[Path]:
    """Return list of canonical .npz files to re-generate."""
    files = sorted(input_dir.glob("*.npz"))
    # Keep only full 14-view files, skip smoke / 4-view / test_set variants to
    # avoid duplicates.  Test set is handled separately.
    return [p for p in files if not _OMIT_RE.search(p.name)]


def _add_fallback_noise(points_2d: np.ndarray, noise_std: float) -> np.ndarray:
    """Add pixel noise to GT 2D points to mimic a real detector (skeleton only)."""
    if noise_std <= 0:
        return points_2d
    noise = np.random.normal(scale=noise_std, size=points_2d.shape).astype(np.float64)
    return points_2d + noise


def _parse_subject_seq_from_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    """Infer subject and sequence numbers from a canonical .npz filename.

    Examples
    --------
    >>> _parse_subject_seq_from_name("s_01_seq_01_v14_multiview.npz")
    (1, 1)
    >>> _parse_subject_seq_from_name("TS1_v14_multiview.npz")
    (None, None)
    """
    m = re.search(r"s_(\d+)_seq_(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _resolve_view_image_paths(
    image_dir: Path,
    npz_name: str,
    frame_idx: int,
    n_views: int,
) -> List[Optional[Path]]:
    """Resolve per-view image paths for a canonical frame.

    Tries several common MPI-INF-3DHP raw-image layouts:

    1. ``image_dir/S{subj}/Seq{seq}/imageSequence/cam{view:02d}/frame_{frame:05d}.jpg``
    2. ``image_dir/S{subj}/Seq{seq}/imageSequence/cam{view:02d}/img_{frame:05d}.png``
    3. ``image_dir/S{subj}/Seq{seq}/video_{view:02d}/frame_{frame:05d}.jpg``

    The first layout that exists for view 0 is assumed for all views.  Missing
    per-view files return ``None`` so the detector can fall back to GT for that
    view.
    """
    subject, seq = _parse_subject_seq_from_name(npz_name)
    if subject is None or image_dir is None or not image_dir.exists():
        return [None] * n_views

    base = image_dir / f"S{subject}" / f"Seq{seq}"
    if not base.exists():
        return [None] * n_views

    candidate_patterns = [
        base / "imageSequence" / f"cam{{view:02d}}" / f"frame_{{frame:05d}}.jpg",
        base / "imageSequence" / f"cam{{view:02d}}" / f"img_{{frame:05d}}.png",
        base / "imageSequence" / f"cam{{view:02d}}" / f"{frame_idx:05d}.jpg",
        base / f"video_{{view:02d}}" / f"frame_{{frame:05d}}.jpg",
    ]

    # Find the first pattern that has an valid image for view 0.
    chosen_pattern: Optional[Path] = None
    for pattern in candidate_patterns:
        candidate = Path(str(pattern).format(view=0, frame=frame_idx))
        if candidate.exists():
            chosen_pattern = pattern
            break

    if chosen_pattern is None:
        return [None] * n_views

    paths: List[Optional[Path]] = []
    for v in range(n_views):
        path = Path(str(chosen_pattern).format(view=v, frame=frame_idx))
        paths.append(path if path.exists() else None)
    return paths


def _generate_detected_sequence(
    input_npz: Path,
    output_npz: Path,
    detector: Detector,
    fallback_noise: float,
    image_dir: Optional[Path],
) -> None:
    """Create a detected-2D copy of one canonical .npz file."""
    data = dict(np.load(input_npz))
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{input_npz} missing keys: {missing}")

    points_2d = data["points_2d"].astype(np.float64)
    confidences = data["confidences"].astype(np.float32)
    joints_3d = data["joints_3d"]
    T, V, J, _ = points_2d.shape

    detected_2d = np.zeros_like(points_2d)
    detected_conf = np.zeros_like(confidences)

    for t in range(T):
        image_paths = _resolve_view_image_paths(image_dir, input_npz.name, t, V)
        d2, dc = detector(image_paths, points_2d[t])
        detected_2d[t] = d2
        detected_conf[t] = dc

    # Fallback / synthetic noise layer.  Once the real detector is wired in,
    # remove this branch.
    if isinstance(detector, FallbackDetector) and fallback_noise > 0:
        detected_2d = _add_fallback_noise(detected_2d, fallback_noise)
        detected_conf = np.clip(detected_conf * 0.9, 0.0, 1.0).astype(np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        points_2d=detected_2d,
        confidences=detected_conf,
        joints_3d=joints_3d,
        camera_K=data["camera_K"],
        camera_R=data["camera_R"],
        camera_t=data["camera_t"],
    )

    print(f"  wrote {output_npz}  shape={detected_2d.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Regenerate MPI-INF-3DHP .npz with detected 2D keypoints."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp"),
        help="Directory containing existing canonical .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp_detected_2d"),
        help="Where to write detected-2D .npz files.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="auto",
        help=(
            "Detector: auto, mediapipe, openpose, fallback. "
            "'auto' tries real detectors and falls back to GT+noise if none are available."
        ),
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Root of raw MPI images (e.g. data/webbridge/mpi_inf_3dhp/raw).",
    )
    parser.add_argument(
        "--fallback_noise",
        type=float,
        default=2.0,
        help="Pixel noise std added to GT 2D when falling back.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the real detector (cuda/cpu).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = _find_source_npz(input_dir)
    if not source_files:
        raise FileNotFoundError(f"No canonical .npz files found in {input_dir}")

    detector = _build_detector(args.detector, device=args.device, fallback_noise=args.fallback_noise)
    if isinstance(detector, FallbackDetector):
        warnings.warn(
            f"Using fallback detector (GT 2D + {args.fallback_noise}px noise). "
            "Install MediaPipe or place OpenPose model files under models/openpose/ "
            "to obtain real detections.",
            stacklevel=1,
        )

    print(f"Generating detected-2D MPI-INF-3DHP .npz files:")
    print(f"  input_dir  : {input_dir.resolve()}")
    print(f"  output_dir : {output_dir.resolve()}")
    print(f"  detector   : {detector.name}")
    print(f"  files      : {len(source_files)}")

    for input_npz in source_files:
        output_npz = output_dir / input_npz.name
        _generate_detected_sequence(
            input_npz,
            output_npz,
            detector,
            fallback_noise=args.fallback_noise,
            image_dir=args.image_dir,
        )

    print("Done. Update configs/splits/mpiinf3dhp_detected_2d.yaml to point here.")


if __name__ == "__main__":
    main()
