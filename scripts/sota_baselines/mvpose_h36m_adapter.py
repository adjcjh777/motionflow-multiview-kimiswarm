#!/usr/bin/env python
"""H36M adapter for zju3dv/mvpose.

This script reads the COCO17-format pickle produced by
``convert_to_mvpose_format.py`` and runs the geometry-only top-down triangulation
kernel from ``zju3dv/mvpose/src/models/estimate3d.py`` to produce per-frame 3D
poses.

Notes
-----
``zju3dv/mvpose`` is an inference-only, multi-person baseline. Its full pipeline
requires a TensorFlow 1.x 2D detector and a PyTorch 1.0 Re-ID backend, which are
not available in the current environment. For H36M (single person, synchronous
multi-view, ground-truth 2D projections), the relevant part of the upstream code
is the geometry-only top-down triangulation in
``MultiEstimator._top_down_pose_kernel``. This adapter therefore bypasses the
2D detector and Re-ID network and drives that kernel directly. If the upstream
repository is not present, it falls back to an equivalent N-view DLT
implementation.

Example
-------
    # CPU smoke test on the first 10 frames of the validation split.
    python scripts/sota_baselines/mvpose_h36m_adapter.py \
        --input_pkl tmp/sota_baselines/mvpose_data_a800/h36m_true_gt_val.pkl \
        --output_dir tmp/sota_baselines/mvpose_predictions \
        --max_frames 10

Output format
-------------
For each sequence in the input pickle, an ``.npz`` file is written containing
``joints_3d`` with shape (F, 17, 3) in the same COCO17 joint order as the input.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dependency handling for zju3dv/mvpose
# ---------------------------------------------------------------------------

_MVPOSE_DIR = Path(__file__).parent.parent.parent / "tmp" / "sota_baselines" / "mvpose"


def _add_mvpose_to_path() -> None:
    if _MVPOSE_DIR.exists() and str(_MVPOSE_DIR) not in sys.path:
        sys.path.insert(0, str(_MVPOSE_DIR))


def _stub_missing_mvpose_modules() -> None:
    """Install lightweight stubs so ``estimate3d`` can be imported without TF/PT1.x."""
    # coloredlogs
    if "coloredlogs" not in sys.modules:
        mod = types.ModuleType("coloredlogs")
        mod.install = lambda *a, **k: None  # type: ignore
        sys.modules["coloredlogs"] = mod

    # backend.CamStyle.reid.utils.data.transforms (used by MemDataset)
    def make_pkg(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore
        sys.modules[name] = m
        return m

    make_pkg("backend")
    make_pkg("backend.CamStyle")
    make_pkg("backend.CamStyle.reid")
    make_pkg("backend.CamStyle.reid.utils")
    make_pkg("backend.CamStyle.reid.utils.data")
    transforms_mod = make_pkg("backend.CamStyle.reid.utils.data.transforms")

    class _T:
        class Normalize:
            def __init__(self, *a, **k):  # noqa: D401
                pass

        class Compose:
            def __init__(self, *a, **k):
                pass

        class Resize:
            def __init__(self, *a, **k):
                pass

        class ToTensor:
            def __init__(self, *a, **k):
                pass

    transforms_mod.Normalize = _T.Normalize  # type: ignore
    transforms_mod.Compose = _T.Compose  # type: ignore
    transforms_mod.Resize = _T.Resize  # type: ignore
    transforms_mod.ToTensor = _T.ToTensor  # type: ignore

    # backend.estimator_2d
    be = types.ModuleType("backend.estimator_2d")

    class Estimator_2d:
        def __init__(self, *a, **k):
            pass

    be.Estimator_2d = Estimator_2d  # type: ignore
    sys.modules["backend.estimator_2d"] = be

    # backend.CamStyle.feature_extract
    bcfe = types.ModuleType("backend.CamStyle.feature_extract")

    class FeatureExtractor:
        def __init__(self, *a, **k):
            pass

        def get_affinity(self, *a, **k):
            return np.ones((4, 4), dtype=np.float32)

    bcfe.FeatureExtractor = FeatureExtractor  # type: ignore
    sys.modules["backend.CamStyle.feature_extract"] = bcfe

    # src.m_lib.pictorial (Cython extension in upstream)
    mlib = types.ModuleType("src.m_lib")
    sys.modules["src.m_lib"] = mlib
    mp = types.ModuleType("src.m_lib.pictorial")
    mp.hybrid_kernel = lambda *a, **k: []  # type: ignore
    mp.getskel = lambda: None  # type: ignore
    mp.getPictoStruct = lambda *a, **k: None  # type: ignore
    mp.inferPict3D_MaxProd = lambda *a, **k: None  # type: ignore
    mp.transform_closure = lambda x: x  # type: ignore
    sys.modules["src.m_lib.pictorial"] = mp


def _try_import_multiestimator() -> Optional[type]:
    """Return ``MultiEstimator`` class if the upstream repo can be imported."""
    if not _MVPOSE_DIR.exists():
        return None
    _add_mvpose_to_path()
    if "src.models.estimate3d" in sys.modules:
        return sys.modules["src.models.estimate3d"].MultiEstimator  # type: ignore
    try:
        _stub_missing_mvpose_modules()
        import src.models.estimate3d as _est  # type: ignore
        return _est.MultiEstimator  # type: ignore
    except Exception as exc:  # pragma: no cover - missing upstream repo/deps
        print(f"[mvpose] Could not import upstream MultiEstimator: {exc}")
        return None


# ---------------------------------------------------------------------------
# Camera / geometry helpers
# ---------------------------------------------------------------------------


def build_projection_matrices(cameras: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build projection matrices from the common camera format.

    Returns
    -------
    P: (V, 3, 4) projection matrices.
    K: (V, 3, 3) intrinsics.
    RT: (V, 3, 4) [R | t] extrinsics.
    """
    Ks, Rs, ts = [], [], []
    for cam in cameras:
        Ks.append(cam["K"])
        Rs.append(cam["R"])
        ts.append(cam["t"])
    K = np.stack(Ks, axis=0).astype(np.float64)
    R = np.stack(Rs, axis=0).astype(np.float64)
    t = np.stack(ts, axis=0).astype(np.float64)  # (V, 3)
    RT = np.concatenate([R, t[:, :, None]], axis=2)  # (V, 3, 4)
    P = K @ RT
    return P, K, RT


def _skew_op(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=np.float64)


def _fundamental_from_rt(
    K0: np.ndarray,
    R0: np.ndarray,
    t0: np.ndarray,
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
) -> np.ndarray:
    """Compute fundamental matrix from two cameras."""
    K0i = np.linalg.inv(K0)
    F = K0i.T @ (R0 @ R1.T) @ K1.T @ _skew_op(K1 @ R1 @ R0.T @ (t0 - R0 @ R1.T @ t1))
    return F


def build_fundamental_matrices(K: np.ndarray, RT: np.ndarray) -> np.ndarray:
    """Build pairwise fundamental matrices F[i, j].

    Matches the implementation in ``src/m_utils/mem_dataset.py``.
    """
    V = K.shape[0]
    F = np.zeros((V, V, 3, 3), dtype=np.float64)
    for i in range(V):
        for j in range(V):
            Ri, ti = RT[i, :, :3], RT[i, :, 3]
            Rj, tj = RT[j, :, :3], RT[j, :, 3]
            F[i, j] = _fundamental_from_rt(K[i], Ri, ti, K[j], Rj, tj)
            if F[i, j].sum() == 0:
                F[i, j] += 1e-12
    return F


def triangulate_dlt_nview(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """N-view DLT triangulation.

    Parameters
    ----------
    P: (V, 3, 4) projection matrices.
    y: (V, 2) image points.

    Returns
    -------
    X: (3,) triangulated point.
    """
    A = []
    for Pi, yi in zip(P, y):
        A.append(yi[0] * Pi[2] - Pi[0])
        A.append(yi[1] * Pi[2] - Pi[1])
    A = np.stack(A, axis=0)  # (2V, 4)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    X = X[:3] / (X[3] + 1e-12)
    return X


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class MVPoseH36MAdapter:
    """Drive ``zju3dv/mvpose`` top-down triangulation on H36M true-GT data."""

    def __init__(self, cameras: List[Dict[str, np.ndarray]], use_mvpose_kernel: bool = True):
        self.P, self.K, self.RT = build_projection_matrices(cameras)
        self.F = build_fundamental_matrices(self.K, self.RT)

        # Default config mirrors model_cfg but forces geometry-only, single-person mode.
        self.cfg = SimpleNamespace(
            use_mincut=False,
            use_bundle=False,
            metric="Geometry only",
            rerank=False,
            hybrid=False,
            spectral=True,
        )

        self.MultiEstimator: Optional[type] = None
        self.est: Optional[Any] = None
        if use_mvpose_kernel:
            self.MultiEstimator = _try_import_multiestimator()
            if self.MultiEstimator is not None:
                self.est = self._make_estimator()

    def _make_estimator(self) -> Any:
        """Create a minimal MultiEstimator instance without 2D/ReID backends."""
        est = self.MultiEstimator.__new__(self.MultiEstimator)  # type: ignore
        est.cfg = self.cfg
        est.dataset = SimpleNamespace(P=self.P)
        est.extractor = SimpleNamespace(get_affinity=lambda *a, **k: np.ones((1, 1)))
        est.est2d = None
        return est

    def _triangulate_mvpose(self, pose2d: np.ndarray) -> Optional[np.ndarray]:
        """Call MultiEstimator._top_down_pose_kernel for a single person/frame."""
        if self.est is None:
            return None

        # Import geometry here to avoid heavy top-level imports.
        from src.m_utils.geometry import (  # type: ignore
            check_bone_length,
            geometry_affinity,
        )

        V = pose2d.shape[0]
        # pose_mat: (V, 17, 2)
        pose_mat = pose2d.astype(np.float32)
        sub_imgid2cam = np.arange(V, dtype=np.int32)
        dimGroup = [0, V]

        geo_affinity_mat = torch.tensor(geometry_affinity(pose_mat, self.F, dimGroup))
        matched_list = [np.arange(V, dtype=np.int32)]
        try:
            multi_pose3d, _ = self.est._top_down_pose_kernel(
                geo_affinity_mat,
                matched_list,
                pose_mat,
                sub_imgid2cam,
            )
        except Exception as exc:  # pragma: no cover - upstream API mismatch
            print(f"[mvpose] _top_down_pose_kernel failed: {exc}")
            return None

        if not multi_pose3d:
            return None

        # multi_pose3d[0] is (3, 17); convert to (17, 3).
        pose3d = multi_pose3d[0].T

        # Reject if the upstream bone-length check failed (returned as empty list).
        if not check_bone_length(pose3d.T):
            return None

        return pose3d

    def estimate_frame(self, pose2d: np.ndarray) -> np.ndarray:
        """Estimate a single 3D pose from multi-view 2D keypoints.

        Parameters
        ----------
        pose2d: (V, 17, 2)

        Returns
        -------
        pose3d: (17, 3)
        """
        if self.est is not None:
            pose3d = self._triangulate_mvpose(pose2d)
            if pose3d is not None:
                return pose3d

        # Fallback: N-view DLT for each joint independently.
        V, J, _ = pose2d.shape
        pose3d = np.empty((J, 3), dtype=np.float64)
        for j in range(J):
            pose3d[j] = triangulate_dlt_nview(self.P, pose2d[:, j, :].astype(np.float64))
        return pose3d

    def __call__(self, points_2d: np.ndarray) -> np.ndarray:
        """Triangulate a batch of frames.

        Parameters
        ----------
        points_2d: (F, V, 17, 2)

        Returns
        -------
        joints_3d: (F, 17, 3)
        """
        F = points_2d.shape[0]
        joints_3d = np.empty((F, 17, 3), dtype=np.float64)
        for i in range(F):
            joints_3d[i] = self.estimate_frame(points_2d[i])
        return joints_3d


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_mvpose_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_predictions(
    output_dir: Path,
    sequence_name: str,
    joints_3d: np.ndarray,
    frame_indices: Optional[np.ndarray] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sequence_name}_pred.npz"
    out: Dict[str, Any] = {"joints_3d": joints_3d}
    if frame_indices is not None:
        out["frame_indices"] = frame_indices
    np.savez_compressed(out_path, **out)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sequence_name(seq: Dict[str, Any], index: int) -> str:
    subject = seq.get("subject", index)
    actions = seq.get("actions", [])
    if actions:
        return f"s_{subject:02d}_acts_{'_'.join(str(a) for a in actions)}"
    return f"s_{subject:02d}_seq_{index}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pkl", type=Path, required=True, help="COCO17-format MVPose pickle.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to write per-sequence predictions.")
    parser.add_argument("--max_frames", type=int, default=None, help="If given, only process this many frames per sequence.")
    parser.add_argument(
        "--use_mvpose_kernel",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use the upstream top-down kernel when available (default: --use-mvpose-kernel).",
    )
    parser.add_argument("--fallback_only", action="store_true", help="Always use the pure-geometry fallback (no upstream import).")
    args = parser.parse_args()

    if not args.input_pkl.exists():
        raise FileNotFoundError(args.input_pkl)

    data = load_mvpose_pickle(args.input_pkl)
    sequences: List[Dict[str, Any]] = data.get("sequences", [])
    print(f"Loaded {len(sequences)} sequence(s) from {args.input_pkl}")

    use_mvpose = args.use_mvpose_kernel and not args.fallback_only

    written: List[Path] = []
    for idx, seq in enumerate(sequences):
        points_2d = seq["points_2d"]  # (F, V, 17, 2)
        cameras = seq["cameras"]

        if args.max_frames is not None:
            points_2d = points_2d[: args.max_frames]
            print(f"Sequence {idx}: processing first {len(points_2d)} frames")
        else:
            print(f"Sequence {idx}: processing all {len(points_2d)} frames")

        adapter = MVPoseH36MAdapter(cameras, use_mvpose_kernel=use_mvpose)
        joints_3d = adapter(points_2d)

        seq_name = _sequence_name(seq, idx)
        out_path = save_predictions(args.output_dir, seq_name, joints_3d)
        written.append(out_path)
        print(f"  wrote {out_path}  shape={joints_3d.shape}")

    # Write a small manifest for downstream evaluation.
    manifest = {
        "input_pkl": str(args.input_pkl),
        "output_dir": str(args.output_dir),
        "num_sequences": len(sequences),
        "predictions": [str(p) for p in written],
        "mvpose_kernel_used": use_mvpose and _try_import_multiestimator() is not None,
    }
    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
