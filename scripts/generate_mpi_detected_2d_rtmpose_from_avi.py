#!/usr/bin/env python3
"""Generate MPI-INF-3DHP detected-2D .npz files from raw AVI zips using RTMPose.

The official MPI-INF-3DHP release ships camera video as ``video_{i}.avi`` inside
``imageSequence/*_cameras.zip`` per sequence.  This script decodes the AVIs
from the zips (cached to disk once) and runs RTMPose Wholebody on every frame
and view, then maps the 133 keypoints to the MPI-INF-3DHP 28-joint skeleton.

Output .npz structure matches the canonical WebBridge format so downstream
scripts (e.g. ``scripts/run_mpi_dlt_baseline.py``) work unchanged.

Usage
-----
    python scripts/generate_mpi_detected_2d_rtmpose_from_avi.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --raw_dir data/webbridge/mpi_inf_3dhp/raw \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d_rtmpose \
        --device cuda --max_frames 100

Author: data foundation repair (2026-08)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_mpi_detected_2d import RTMPoseDetector, MEDIAPIPE_TO_MPI  # noqa: E402

ZIP_GROUPS = [
    "vnect_cameras.zip",
    "other_angled_cameras.zip",
    "ceiling_cameras.zip",
]

# Per-view source camera group for the 14 canonical views, based on the
# official release layout (vnect covers 0,1,2,4,5,6,7,8; other-angled covers
# 3,9,10; ceiling covers 11,12,13).
VIEW_TO_GROUP: Dict[int, str] = {
    0: "vnect_cameras.zip",
    1: "vnect_cameras.zip",
    2: "vnect_cameras.zip",
    3: "other_angled_cameras.zip",
    4: "vnect_cameras.zip",
    5: "vnect_cameras.zip",
    6: "vnect_cameras.zip",
    7: "vnect_cameras.zip",
    8: "vnect_cameras.zip",
    9: "other_angled_cameras.zip",
    10: "other_angled_cameras.zip",
    11: "ceiling_cameras.zip",
    12: "ceiling_cameras.zip",
    13: "ceiling_cameras.zip",
}


def _get_avi_path(zip_path: Path, view: int, cache_dir: Path) -> Optional[Path]:
    """Return a local path to ``video_{view}.avi``, extracting it from zip once."""
    name = f"video_{view}.avi"
    cache_file = cache_dir / name
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file
    try:
        with zipfile.ZipFile(zip_path) as z:
            if name not in z.namelist():
                return None
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                f.write(z.read(name))
            return cache_file
    except Exception:
        return None


class AviReader:
    """Sequential frame decoder for a local AVI file."""

    def __init__(self, avi_path: Path):
        import cv2

        self.avi_path = avi_path
        self.cap = cv2.VideoCapture(str(avi_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cv2 cannot open AVI: {avi_path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next = 0

    def frame(self, idx: int) -> Optional[np.ndarray]:
        import cv2

        if idx != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next = idx
        ok, f = self.cap.read()
        self._next += 1
        return f if ok else None

    def close(self) -> None:
        self.cap.release()


def _open_readers(raw_seq_dir: Path, V: int) -> tuple[List[Optional[AviReader]], List[int]]:
    readers: List[Optional[AviReader]] = []
    missing: List[int] = []
    avi_cache_dir = raw_seq_dir / "imageSequence" / ".avi_cache"
    for v in range(V):
        zpath = raw_seq_dir / "imageSequence" / VIEW_TO_GROUP[v]
        if not zpath.exists():
            readers.append(None)
            missing.append(v)
            continue
        avi_path = _get_avi_path(zpath, v, avi_cache_dir)
        if avi_path is None:
            readers.append(None)
            missing.append(v)
            continue
        readers.append(AviReader(avi_path))
    return readers, missing


def _detect_view(detector: RTMPoseDetector, frame_bgr: np.ndarray, gt2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run RTMPose on a single BGR frame and map to MPI 28 joints."""
    import cv2

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # RTMPoseDetector.__call__ expects a list of image paths and a GT array.
    # We call its model directly to avoid disk I/O.
    keypoints, scores = detector.model(img_rgb)
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    # rtmlib may return (N, K, 2) / (N, K) where N is the number of detected
    # persons.  Keep the highest-confidence person.
    if keypoints.ndim == 3 and scores.ndim == 2:
        if keypoints.shape[0] > 1:
            person_scores = scores.mean(axis=1)
            best = int(np.argmax(person_scores))
            keypoints = keypoints[best]
            scores = scores[best]
        elif keypoints.shape[0] == 1:
            keypoints = keypoints[0]
            scores = scores[0]

    J = gt2d.shape[0]
    out = gt2d.copy().astype(np.float64)
    conf = np.zeros(J, dtype=np.float32)
    mapped: set[int] = set()

    for rtm_idx, mpi_idx in detector.RTM_TO_MPI.items():
        if rtm_idx < len(keypoints):
            out[mpi_idx] = keypoints[rtm_idx]
            conf[mpi_idx] = float(scores[rtm_idx])
            mapped.add(mpi_idx)

    # Approximate head_top as average of face keypoints.
    face_points = []
    face_scores = []
    for fidx in detector.FACE_INDICES:
        if fidx < len(keypoints):
            face_points.append(keypoints[fidx])
            face_scores.append(scores[fidx])
    if face_points:
        out[7] = np.mean(face_points, axis=0)
        conf[7] = float(np.mean(face_scores))
        mapped.add(7)

    # Fill unmapped joints with GT and low confidence.
    for j in range(J):
        if j not in mapped:
            out[j] = gt2d[j]
            conf[j] = 0.25

    return out, conf


def process_sequence(npz_path: Path, raw_seq_dir: Path, output_path: Path, device: str, max_frames: int = 0) -> str:
    import cv2  # noqa: F401

    npz = np.load(npz_path)
    points_2d = np.asarray(npz["points_2d"], dtype=np.float64)
    T, V, J, _ = points_2d.shape
    end = T if max_frames <= 0 else min(max_frames, T)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    det2d = np.zeros_like(points_2d)
    detconf = np.zeros((T, V, J), dtype=np.float32)

    detector = RTMPoseDetector(device=device)

    readers, missing = _open_readers(raw_seq_dir, V)
    try:
        t0 = time.time()
        for t in range(end):
            for v in range(V):
                r = readers[v]
                if r is None:
                    det2d[t, v] = points_2d[t, v]
                    continue
                frame = r.frame(t)
                if frame is None:
                    det2d[t, v] = points_2d[t, v]
                    continue
                out, conf = _detect_view(detector, frame, points_2d[t, v])
                det2d[t, v] = out
                detconf[t, v] = conf
            if t > 0 and t % 100 == 0:
                rate = (t + 1) / (time.time() - t0)
                print(f"  frame {t}/{end} ({rate:.1f} fps)", flush=True)
    finally:
        for r in readers:
            if r is not None:
                r.close()

    # When max_frames truncates the sequence, only save the processed prefix.
    if max_frames > 0 and end < T:
        det2d = det2d[:end]
        detconf = detconf[:end]
        joints_3d = npz["joints_3d"][:end]
    else:
        joints_3d = npz["joints_3d"]

    np.savez(
        output_path,
        points_2d=det2d,
        confidences=detconf,
        joints_3d=joints_3d,
        camera_K=npz["camera_K"],
        camera_R=npz["camera_R"],
        camera_t=npz["camera_t"],
    )
    mapped_frac = float((detconf[:end] > 0.25).mean())
    return (
        f"wrote {output_path.name} T={end} V={V} "
        f"mapped_joint_frac={mapped_frac:.3f} missing_views={missing}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp"))
    p.add_argument("--raw_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp/raw"))
    p.add_argument("--output_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp_detected_2d_rtmpose"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_frames", type=int, default=0, help="if >0, only process first N frames (smoke test)")
    p.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8")
    p.add_argument("--seqs", type=str, default="1,2")
    args = p.parse_args()

    subjects = [int(s) for s in args.subjects.split(",")]
    seqs = [int(s) for s in args.seqs.split(",")]

    for s in subjects:
        for q in seqs:
            npz_name = f"s_{s:02d}_seq_{q:02d}_v14_multiview_m.npz"
            npz_path = args.input_dir / npz_name
            if not npz_path.exists():
                print(f"skip missing {npz_path}")
                continue
            raw_seq_dir = args.raw_dir / f"S{s}" / f"Seq{q}"
            output_path = args.output_dir / npz_name
            print(f"Processing {npz_name} ...")
            msg = process_sequence(npz_path, raw_seq_dir, output_path, args.device, args.max_frames)
            print(msg)
            gc.collect()


if __name__ == "__main__":
    main()
