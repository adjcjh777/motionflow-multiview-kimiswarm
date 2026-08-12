#!/usr/bin/env python3
"""Generate MPI-INF-3DHP detected-2D .npz files from raw AVI zips using RTMPose.

Mirrors ``generate_mpi_detected_2d_from_avi.py`` but uses the RTMPose Wholebody
model via ``rtmlib`` instead of MediaPipe.  The raw MPI-INF-3DHP release ships
camera video as ``video_{i}.avi`` inside ``imageSequence/*_cameras.zip``; this
script decodes frames directly from those zips without extracting the full
~7 GB of frames to disk.

Usage
-----
    python scripts/generate_mpi_rtmpose_from_avi.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --raw_dir data/webbridge/mpi_inf_3dhp/raw \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d_rtmpose \
        --subjects 1 --seqs 1 --max_frames 50

Author: research swarm (data foundation repair, 2026-08)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# RTMPose Wholebody -> MPI-INF-3DHP 28-joint mapping
# ---------------------------------------------------------------------------
RTM_TO_MPI = {
    0: 6,    # nose            -> head
    5: 9,    # left_shoulder   -> left_shoulder
    6: 14,   # right_shoulder  -> right_shoulder
    7: 10,   # left_elbow      -> left_elbow
    8: 15,   # right_elbow     -> right_elbow
    9: 11,   # left_wrist      -> left_wrist
    10: 16,  # right_wrist     -> right_wrist
    11: 18,  # left_hip        -> left_hip
    12: 23,  # right_hip       -> right_hip
    13: 19,  # left_knee       -> left_knee
    14: 24,  # right_knee      -> right_knee
    15: 20,  # left_ankle      -> left_ankle
    16: 25,  # right_ankle     -> right_ankle
    17: 22,  # left_big_toe    -> left_toe
    20: 21,  # left_heel       -> left_foot
    21: 27,  # right_big_toe   -> right_toe
    22: 26,  # right_heel      -> right_foot
}

FACE_INDICES = [1, 2, 3, 4, 68, 69, 70, 71, 72, 73, 74, 75]

ZIP_GROUPS = [
    "vnect_cameras.zip",
    "other_angled_cameras.zip",
    "ceiling_cameras.zip",
]

# Per-view source camera group for the 14 canonical views.
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
    """Return a local path to ``video_{view}.avi``, extracting it from *zip_path* once."""
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
            raise RuntimeError("cv2 cannot open AVI (ffmpeg backend missing?)")
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


def _rtmpose_detect(model, frame_rgb: np.ndarray, gt2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run RTMPose Wholebody on a single RGB frame and map to MPI 28 joints.

    Returns (points_2d, confidences) for the 28-joint skeleton. Unmapped
    joints are filled from the GT 2D projection with confidence 0.0.
    """
    J = gt2d.shape[0]
    out = gt2d.copy().astype(np.float64)
    conf = np.zeros(J, dtype=np.float32)

    keypoints, scores = model(frame_rgb)
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    # rtmlib may return (N, K, 2) when it detects N people. Keep the first
    # detection (highest confidence) for each keypoint.
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]

    # Sanity check: should now have one keypoint per score.
    if keypoints.shape[0] != scores.shape[0]:
        raise RuntimeError(
            f"RTMPose output shape mismatch: keypoints {keypoints.shape}, scores {scores.shape}"
        )

    mapped: set[int] = set()
    for rtm_idx, mpi_idx in RTM_TO_MPI.items():
        if rtm_idx < len(keypoints):
            out[mpi_idx] = keypoints[rtm_idx]
            conf[mpi_idx] = float(scores[rtm_idx])
            mapped.add(mpi_idx)

    # Approximate head_top as the average of available face keypoints.
    face_points = []
    face_scores = []
    for fidx in FACE_INDICES:
        if fidx < len(keypoints):
            face_points.append(keypoints[fidx])
            face_scores.append(scores[fidx])
    if face_points:
        out[7] = np.mean(face_points, axis=0)
        conf[7] = float(np.mean(face_scores))
        mapped.add(7)

    # Unmapped joints keep GT 2D with zero confidence (explicitly not detected).
    for j in range(J):
        if j not in mapped:
            out[j] = gt2d[j]
            conf[j] = 0.0
    return out, conf


def _load_or_init_arrays(out_path: Path, points_2d: np.ndarray, T: int, V: int, J: int):
    """Load an existing partial output or allocate fresh zero arrays."""
    if out_path.exists():
        try:
            existing = np.load(out_path)
            det2d = np.asarray(existing["points_2d"], dtype=np.float64)
            detconf = np.asarray(existing["confidences"], dtype=np.float32)
            if det2d.shape == points_2d.shape and detconf.shape == (T, V, J):
                return det2d, detconf
        except Exception:
            pass
    return np.zeros_like(points_2d), np.zeros((T, V, J), dtype=np.float32)


def process_file(args_ns, npz_path: Path) -> str:
    """Generate one detected-2D npz using RTMPose."""
    import cv2  # noqa: F401

    npz = np.load(npz_path)
    points_2d = np.asarray(npz["points_2d"], dtype=np.float64)
    T, V, J, _ = points_2d.shape

    raw_seq_dir = args_ns.raw_dir / f"S{args_ns.subject}" / f"Seq{args_ns.seq}"
    out_path = args_ns.output_dir / npz_path.name
    avi_cache_dir = raw_seq_dir / "imageSequence" / ".avi_cache"
    det2d, detconf = _load_or_init_arrays(out_path, points_2d, T, V, J)

    end = T if args_ns.max_frames <= 0 else min(args_ns.max_frames, T)

    from rtmlib import Wholebody

    det = Wholebody(mode=args_ns.mode, device=args_ns.device, backend="onnxruntime")

    readers: List[Optional[AviReader]] = []
    missing_views: List[int] = []
    for v in range(V):
        zpath = raw_seq_dir / "imageSequence" / VIEW_TO_GROUP[v]
        if not zpath.exists():
            readers.append(None)
            missing_views.append(v)
            continue
        avi_path = _get_avi_path(zpath, v, avi_cache_dir)
        if avi_path is None:
            readers.append(None)
            missing_views.append(v)
            continue
        readers.append(AviReader(avi_path))

    n_frames_missing_det = 0
    t0 = time.time()
    for t in range(end):
        for v in range(V):
            r = readers[v]
            if r is None:
                det2d[t, v] = points_2d[t, v]
                detconf[t, v] = 0.0
                continue
            frame_bgr = r.frame(t)
            if frame_bgr is None:
                n_frames_missing_det += 1
                det2d[t, v] = points_2d[t, v]
                detconf[t, v] = 0.0
                continue
            # RTMPose Wholebody expects RGB numpy array.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            d2, dc = _rtmpose_detect(det, frame_rgb, points_2d[t, v])
            det2d[t, v] = d2
            detconf[t, v] = dc
        if t > 0 and t % 100 == 0:
            rate = (t + 1) / (time.time() - t0)
            print(
                f"  S{args_ns.subject}/Seq{args_ns.seq} frame {t}/{end} "
                f"({rate:.1f} frames/s)",
                flush=True,
            )

    for r in readers:
        if r is not None:
            r.close()

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
        f"processed_frames={end} mapped_joint_frac={mapped_frac:.3f} "
        f"missing_view_list={missing_views} frames_without_any_det={n_frames_missing_det} "
        f"elapsed={time.time()-t0:.0f}s"
    )
    return msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp"))
    p.add_argument("--raw_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp/raw"))
    p.add_argument("--output_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp_detected_2d_rtmpose"))
    p.add_argument("--subjects", type=str, default="1,3,4,5,6,7,8,2",
                   help="comma-separated subject ids to process")
    p.add_argument("--seqs", type=str, default="1,2")
    p.add_argument("--only_m", action="store_true", default=True,
                   help="only process metre-convention *_m.npz files")
    p.add_argument("--max_frames", type=int, default=0,
                   help="if >0, only process the first N frames (smoke test)")
    p.add_argument("--device", type=str, default="cpu",
                   help="device passed to rtmlib Wholebody (cpu/cuda)")
    p.add_argument("--mode", type=str, default="balanced",
                   help="rtmlib Wholebody mode (lightweight/balanced/performance)")
    p.add_argument("--skip_existing", action="store_true", default=False,
                   help="skip sequences whose output .npz already exists")
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
            out_npz = args.output_dir / name
            if args.skip_existing and out_npz.exists():
                print(f"skip (output already exists): {out_npz}", flush=True)
                continue
            jobs.append((s, q, npz))

    print(f"processing {len(jobs)} sequences", flush=True)

    for s, q, npz in jobs:
        ns = argparse.Namespace(**vars(args), subject=s, seq=q)
        try:
            print(process_file(ns, npz), flush=True)
        except Exception as exc:
            warnings.warn(f"S{s}/Seq{q} FAILED: {exc}")
        gc.collect()


if __name__ == "__main__":
    main()
