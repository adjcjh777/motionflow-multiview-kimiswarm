#!/usr/bin/env python3
"""Generate MPI-INF-3DHP detected-2D .npz files from the raw vnect/other-angled/
ceiling camera AVI zips using MediaPipe Pose (tasks API).

Why a new script instead of ``generate_mpi_detected_2d.py``:
* The official MPI-INF-3DHP release ships camera video as ``video_{i}.avi``
  inside ``imageSequence/*_cameras.zip`` per sequence — not extracted jpg
  frames.  This script decodes the AVIs directly from the zips (no disk
  extraction of ~7 GB of frames).
* mediapipe >= 1.0 removed the legacy ``mp.solutions.pose`` API used by the
  older script; this one uses the tasks ``PoseLandmarker`` API.

Protocol (P0-2, issue #191):
* Input canonical npz: ``data/webbridge/mpi_inf_3dhp/s_XX_seq_YY_v14_multiview*.npz``
  (GT-projected 2D kept only for joints MediaPipe does not model — those get
  confidence 0.0, so downstream gates/models ignore them).
* Output: same keys, ``points_2d`` replaced with MediaPipe detections for the
  mapped joints, ``confidences`` = MediaPipe ``visibility`` (mapped joints) /
  0.0 (unmapped joints, value kept from GT 2D).
* Camera groups per sequence (official release):
  - ``vnect_cameras.zip``          -> video_0..video_8 except 3 (i.e. 0,1,2,4,5,6,7,8)
  - ``other_angled_cameras.zip``   -> video_3, video_9, video_10
  - ``ceiling_cameras.zip``        -> video_11, video_12, video_13
* Frame alignment: AVI frame t == canonical npz frame t (verified on S1/Seq1:
  both have 6416 frames at 25 fps).

Usage
-----
    python scripts/generate_mpi_detected_2d_from_avi.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --raw_dir data/webbridge/mpi_inf_3dhp/raw \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
        --model models/mediapipe/pose_landmarker_full.task \
        --detect_size 512 --workers 4

Author: research swarm (data foundation repair, 2026-08)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the audited MediaPipe -> MPI-INF-3DHP joint mapping.
from scripts.generate_mpi_detected_2d import MEDIAPIPE_TO_MPI  # noqa: E402

ZIP_GROUPS = [
    "vnect_cameras.zip",
    "other_angled_cameras.zip",
    "ceiling_cameras.zip",
]

# Per-view source camera group for the 14 canonical views, based on the
# official release layout (vnect covers 0,1,2,4,5,6,7,8; other-angled covers
# 3,9,10; ceiling covers 11,12,13).
VIEW_TO_GROUP: Dict[int, str] = {
    0: "vnect_cameras.zip", 1: "vnect_cameras.zip", 2: "vnect_cameras.zip",
    3: "other_angled_cameras.zip",
    4: "vnect_cameras.zip", 5: "vnect_cameras.zip", 6: "vnect_cameras.zip",
    7: "vnect_cameras.zip", 8: "vnect_cameras.zip",
    9: "other_angled_cameras.zip", 10: "other_angled_cameras.zip",
    11: "ceiling_cameras.zip", 12: "ceiling_cameras.zip",
    13: "ceiling_cameras.zip",
}

# Joints that are filled by more than one MediaPipe landmark (averaged).
MULTI_SOURCE_MPI = {mpi for lst in [MEDIAPIPE_TO_MPI.values()] for mpi in lst if list(MEDIAPIPE_TO_MPI.values()).count(mpi) > 1}


def _extract_avi_bytes(zip_path: Path, view: int) -> Optional[bytes]:
    """Return the raw bytes of ``video_{view}.avi`` inside *zip_path*."""
    name = f"video_{view}.avi"
    try:
        with zipfile.ZipFile(zip_path) as z:
            if name not in z.namelist():
                return None
            return z.read(name)
    except Exception:
        return None


class AviReader:
    """Sequential frame decoder for an in-memory AVI (written to a temp file)."""

    def __init__(self, avi_bytes: bytes):
        import cv2

        fd, tmp = tempfile.mkstemp(suffix=".avi")
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(avi_bytes)
        self.tmp = tmp
        self.cap = cv2.VideoCapture(tmp)
        if not self.cap.isOpened():
            raise RuntimeError("cv2 cannot open AVI (ffmpeg backend missing?)")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame(self, idx: int) -> Optional[np.ndarray]:
        import cv2

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        return f if ok else None

    def close(self) -> None:
        self.cap.release()
        try:
            os.remove(self.tmp)
        except OSError:
            pass


class MediaPipeTasksDetector:
    """MediaPipe PoseLandmarker (tasks API) single-pose detector."""

    def __init__(self, model_path: str, detect_size: int = 512):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self.mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.size = int(detect_size)

    def detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return (33, 3) array of [x_px, y_px, visibility] at ORIGINAL scale."""
        import cv2

        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.size, self.size))
        img = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
        )
        res = self.landmarker.detect(img)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        out = np.zeros((33, 3), dtype=np.float64)
        for i, l in enumerate(lm):
            out[i, 0] = l.x * w
            out[i, 1] = l.y * h
            out[i, 2] = float(l.visibility)
        return out


def map_landmarks_to_mpi(
    lm: Optional[np.ndarray], gt2d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Map a (33,3) MediaPipe detection onto the MPI 28-joint layout.

    Returns (points_2d (J,2), confidences (J,)).  Unmapped joints keep the
    GT 2D value with confidence 0.0 (explicitly 'not detected').
    """
    J = gt2d.shape[0]
    out = gt2d.copy().astype(np.float64)
    conf = np.zeros(J, dtype=np.float32)
    if lm is None:
        return out, conf

    acc: Dict[int, List[np.ndarray]] = {}
    acc_v: Dict[int, List[float]] = {}
    for mp_idx, mpi_idx in MEDIAPIPE_TO_MPI.items():
        acc.setdefault(mpi_idx, []).append(lm[mp_idx, :2])
        acc_v.setdefault(mpi_idx, []).append(float(lm[mp_idx, 2]))
    for mpi_idx in acc:
        out[mpi_idx] = np.mean(acc[mpi_idx], axis=0)
        conf[mpi_idx] = float(np.max(acc_v[mpi_idx]))
    return out, conf


def process_file(args_ns, npz_path: Path) -> str:
    """Generate one detected-2D npz. Runs inside a worker process."""
    import cv2  # noqa: F401  (ensure import in child)

    npz = np.load(npz_path)
    points_2d = np.asarray(npz["points_2d"], dtype=np.float64)
    T, V, J, _ = points_2d.shape

    raw_seq_dir = args_ns.raw_dir / f"S{args_ns.subject}" / f"Seq{args_ns.seq}"
    det = MediaPipeTasksDetector(args_ns.model, args_ns.detect_size)

    # Open all 14 cameras.
    readers: List[Optional[AviReader]] = []
    missing_views: List[int] = []
    for v in range(V):
        zpath = raw_seq_dir / "imageSequence" / VIEW_TO_GROUP[v]
        if not zpath.exists():
            readers.append(None)
            missing_views.append(v)
            continue
        avi = _extract_avi_bytes(zpath, v)
        if avi is None:
            readers.append(None)
            missing_views.append(v)
            continue
        readers.append(AviReader(avi))

    det2d = np.zeros_like(points_2d)
    detconf = np.zeros((T, V, J), dtype=np.float32)
    n_frames_missing_det = 0
    t0 = time.time()
    Tmax = args_ns.max_frames if getattr(args_ns, "max_frames", 0) > 0 else T
    for t in range(Tmax):
        for v in range(V):
            r = readers[v]
            if r is None:
                det2d[t, v] = points_2d[t, v]
                detconf[t, v] = 0.0
                continue
            frame = r.frame(t)
            lm = det.detect(frame) if frame is not None else None
            if lm is None:
                n_frames_missing_det += 1
            p, c = map_landmarks_to_mpi(lm, points_2d[t, v])
            det2d[t, v] = p
            detconf[t, v] = c
        if t > 0 and t % 500 == 0:
            rate = t / (time.time() - t0)
            print(
                f"  S{args_ns.subject}/Seq{args_ns.seq} frame {t}/{T} "
                f"({rate:.1f} frames/s)",
                flush=True,
            )

    for r in readers:
        if r is not None:
            r.close()

    out_path = args_ns.output_dir / npz_path.name
    np.savez(
        out_path,
        points_2d=det2d,
        confidences=detconf,
        joints_3d=npz["joints_3d"],
        camera_K=npz["camera_K"],
        camera_R=npz["camera_R"],
        camera_t=npz["camera_t"],
    )
    mapped_frac = float((detconf > 0).mean())
    msg = (
        f"S{args_ns.subject}/Seq{args_ns.seq}: wrote {out_path.name} T={T} V={V} "
        f"mapped_joint_frac={mapped_frac:.3f} missing_view_list={missing_views} "
        f"frames_without_any_det={n_frames_missing_det} "
        f"elapsed={time.time()-t0:.0f}s"
    )
    return msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp"))
    p.add_argument("--raw_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp/raw"))
    p.add_argument("--output_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp_detected_2d"))
    p.add_argument("--model", type=str, default="models/mediapipe/pose_landmarker_full.task")
    p.add_argument("--detect_size", type=int, default=512)
    p.add_argument("--subjects", type=str, default="1,3,4,5,6,7,8,2",
                   help="comma-separated subject ids to process")
    p.add_argument("--seqs", type=str, default="1,2")
    p.add_argument("--only_m", action="store_true", default=True,
                   help="only process metre-convention *_m.npz files")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel sequences (each uses one MediaPipe instance)")
    p.add_argument("--max_frames", type=int, default=0,
                   help="if >0, only process the first N frames (smoke test)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = [int(s) for s in args.subjects.split(",")]
    seqs = [int(s) for s in args.seqs.split(",")]

    jobs = []
    for s in subjects:
        for q in seqs:
            stem = f"s_{s:02d}_seq_{q:02d}_v14_multiview"
            name = f"{stem}_m.npz" if args.only_m else f"{stem}.npz"
            npz = args.input_dir / name
            if not npz.exists():
                print(f"skip (missing canonical npz): {npz}", flush=True)
                continue
            jobs.append((s, q, npz))

    print(f"processing {len(jobs)} sequences with {args.workers} workers", flush=True)

    if args.workers <= 1:
        for s, q, npz in jobs:
            ns = argparse.Namespace(**vars(args), subject=s, seq=q)
            print(process_file(ns, npz), flush=True)
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for s, q, npz in jobs:
            ns = argparse.Namespace(**vars(args), subject=s, seq=q)
            futs[ex.submit(process_file, ns, npz)] = (s, q)
        for fut in as_completed(futs):
            s, q = futs[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as exc:
                warnings.warn(f"S{s}/Seq{q} FAILED: {exc}")


if __name__ == "__main__":
    main()
