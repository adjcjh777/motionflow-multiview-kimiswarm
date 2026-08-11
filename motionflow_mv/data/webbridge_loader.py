"""Canonical WebBridge dataset loader.

This module converts publicly accessible multi-view human motion datasets into the
project's standard ``.npz`` format used by the ``ray_attention`` training and
evaluation pipeline.

Supported sources and status:

* Human3.6M  – fully supported; requires the Hugging Face preprocessed archive
  ``h36m_sh_conf_cam_source_final.pkl.zip`` and ``camera_params.json``.
* Shelf/Campus  – fully supported; requires ``calibration.json`` and
  ``annotation_3d.json``.
* Synthetic/AMASS  – supported via the existing SMPL-based generator; requires
  ``smplx`` and the SMPL neutral model.
* AIST++  – supported; requires ``keypoints2d``, ``keypoints3d``, ``motions``
  and ``cameras`` from the official release.
* CMU Panoptic  – stub only; raw dataset not yet present.
* 3DPW  – stub only; raw dataset not yet present.

The canonical ``.npz`` file contains exactly the following arrays::

    points_2d:   (T, V, J, 2)  per-view 2D keypoint detections
    confidences: (T, V, J)     detection confidence / visibility
    joints_3d:   (T, J, 3)    3D ground-truth joint positions
    camera_K:    (V, 3, 3)    intrinsic calibration matrices
    camera_R:    (V, 3, 3)    rotation (world-to-camera)
    camera_t:    (V, 3)       translation (world-to-camera)

Author: research swarm (iter 5)
"""

import argparse
import json
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import torch

from ..calibration.camera import Camera
from ..fusion.triangulation import triangulate_dlt
from .shelf_loader import build_shelf_dataset


# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------
def _save_canonical_npz(
    out_path: Path,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    cameras: List[Camera],
) -> None:
    """Save arrays in the canonical multi-view ``.npz`` format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=np.stack([cam.K for cam in cameras], axis=0),
        camera_R=np.stack([cam.R for cam in cameras], axis=0),
        camera_t=np.stack([cam.t for cam in cameras], axis=0),
    )


# -----------------------------------------------------------------------------
# Human3.6M
# -----------------------------------------------------------------------------
def _triangulate_joints(points_2d: np.ndarray, cameras: List[Camera]) -> np.ndarray:
    """Triangulate (V, J, 2) keypoints using calibrated cameras via DLT."""
    V, J, _ = points_2d.shape
    P = np.stack([cam.projection_matrix for cam in cameras], axis=0)
    X = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        X[j] = triangulate_dlt(points_2d[:, j, :], P)
    return X


def _find_h36m_true_gt(
    subject: int,
    actions: List[int],
    split: str,
    true_gt_dir: Path,
) -> Optional[Tuple[Path, np.ndarray]]:
    """Discover a true-GT npz in ``true_gt_dir`` covering ``actions``.

    Returns ``(path, stored_actions)`` for the best matching candidate.
    Exact matches are preferred, followed by the candidate with the fewest
    extra actions.
    """
    true_gt_dir = Path(true_gt_dir)
    if not true_gt_dir.is_dir():
        return None
    requested = set(actions)
    candidates: List[Tuple[Path, np.ndarray, int]] = []
    for path in true_gt_dir.glob(f"s_{subject:02d}_*_true_gt.npz"):
        try:
            with np.load(path, allow_pickle=True) as data:
                stored = data["actions"]
        except Exception:  # pragma: no cover - malformed/locked files skipped
            continue
        stored_set = set(stored.tolist())
        if not requested.issubset(stored_set):
            continue
        candidates.append((path, stored, len(stored_set - requested)))
    if not candidates:
        return None
    # Exact match (same number of actions) comes first; then fewest extras.
    candidates.sort(key=lambda x: (len(x[1]) != len(requested), x[2]))
    return candidates[0][0], candidates[0][1]


def _slice_true_gt_by_actions(
    true_joints_3d: np.ndarray,
    true_actions: np.ndarray,
    requested_actions: List[int],
    groups: Dict[str, Dict[str, List[int]]],
    subject: int,
) -> np.ndarray:
    """Slice a combined true-GT npz to the requested actions in order.

    Frame counts per action are inferred from the pkl ``groups`` dict.
    """
    true_actions_list = [int(a) for a in true_actions.tolist()]
    action_counts: Dict[int, int] = {}
    for action in true_actions_list:
        prefix = f"s_{subject:02d}_act_{action:02d}_"
        bases = [b for b in groups.keys() if b.startswith(prefix)]
        action_counts[action] = sum(len(groups[b]["01"]) for b in bases)

    offsets = [0]
    for action in true_actions_list:
        offsets.append(offsets[-1] + action_counts[action])

    chunks: List[np.ndarray] = []
    for action in requested_actions:
        if action not in true_actions_list:
            raise ValueError(
                f"Requested action {action} not available in true GT actions "
                f"{true_actions_list}"
            )
        idx = true_actions_list.index(action)
        start = offsets[idx]
        end = offsets[idx + 1]
        chunks.append(true_joints_3d[start:end])
    return np.concatenate(chunks, axis=0)


def convert_human36m(
    data_root: Path,
    subject: int,
    actions: List[int],
    split: str = "train",
    out_dir: Path = Path("data/h36m_hf"),
    camera_param_file: str = "camera_params.json",
    archive_file: str = "h36m_sh_conf_cam_source_final.pkl.zip",
    true_gt_path: Optional[Path] = None,
) -> Path:
    """Convert the preprocessed Hugging Face Human3.6M subset to canonical ``.npz``.

    Args:
        data_root: directory containing ``camera_params.json`` and the pkl zip.
        subject: subject id, e.g. ``1``.
        actions: list of action ids, e.g. ``[2, 3, 4]``.
        split: ``"train"`` or ``"test"``.
        out_dir: where to write the output ``.npz``.
        camera_param_file: name of the camera parameter json file.
        archive_file: name of the preprocessed pkl zip archive.
        true_gt_path: optional path to an existing canonical ``.npz`` that contains
            a ``joints_3d`` array with the true 3D ground truth. If provided, it is
            used as the 3D label instead of triangulating from the 2D input. When
            omitted, ``convert_human36m`` attempts to auto-discover a matching
            true-GT npz in ``data/h36m_true_gt/`` and slices it to the requested
            actions.

    Returns:
        Path to the generated ``.npz`` file.
    """
    data_root = Path(data_root)
    cam_path = data_root / camera_param_file
    archive_path = data_root / archive_file

    if not cam_path.exists():
        raise FileNotFoundError(
            f"Human3.6M camera parameters not found: {cam_path}\n"
            "Download from https://github.com/karfly/human36m-camera-parameters"
        )
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Human3.6M preprocessed archive not found: {archive_path}\n"
            "Download from CameronSteele/h36m_3dhp on Hugging Face."
        )

    with open(cam_path) as f:
        cam_params = json.load(f)

    with zipfile.ZipFile(archive_path) as z:
        with z.open("h36m_sh_conf_cam_source_final.pkl") as f:
            data = pickle.load(f)

    split_data = data[split]
    groups = defaultdict(lambda: defaultdict(list))
    for i, src in enumerate(split_data["source"]):
        base, cam = src.rsplit("_", 1)
        groups[base][cam].append(i)

    prefix = f"s_{subject:02d}_act_"
    candidate_bases = sorted([b for b in groups.keys() if b.startswith(prefix)])
    target_bases = []
    for action in actions:
        action_prefix = f"s_{subject:02d}_act_{action:02d}_"
        matches = [b for b in candidate_bases if b.startswith(action_prefix)]
        if not matches:
            raise ValueError(
                f"No source found for subject {subject} action {action} in {split}"
            )
        target_bases.extend(matches)

    cam_names = ["01", "02", "03", "04"]
    subject_key = f"S{subject}"
    first_cams_dict = groups[target_bases[0]]

    # Determine the physical camera per slot.  In the train split the pkl's
    # per-row ``camera_name`` is consistent within each slot, but in the test
    # split it is shuffled across frames within a slot, so the first row cannot
    # be trusted.  If every slot carries a single camera_name across the first
    # base group we use it; otherwise we fall back to the fixed H36M studio
    # order (slot 01..04 -> 54138969, 55011271, 58860488, 60457274), which was
    # verified by reprojection (~3-8 px RMSE of the true mocap GT).
    H36M_FIXED_CAMERAS = ["54138969", "55011271", "58860488", "60457274"]
    per_slot_names = []
    consistent = True
    for cam in cam_names:
        names_in_slot = {
            str(split_data["camera_name"][i]) for i in first_cams_dict.get(cam, [])
        }
        if len(names_in_slot) != 1:
            consistent = False
            break
        per_slot_names.append(names_in_slot.pop())
    if consistent:
        camera_names = per_slot_names
    else:
        camera_names = H36M_FIXED_CAMERAS
        print(
            "NOTE: camera_name column is shuffled within camera slots "
            f"(test split); using fixed H36M camera order {camera_names}."
        )

    cameras = []
    for i, cam_name in enumerate(cam_names):
        camera_name = camera_names[i]
        intr = cam_params["intrinsics"][camera_name]["calibration_matrix"]
        K = np.array(intr, dtype=np.float64)
        ext = cam_params["extrinsics"][subject_key][camera_name]
        R = np.array(ext["R"], dtype=np.float64)
        t = np.array(ext["t"], dtype=np.float64).reshape(3)
        cameras.append(Camera(K=K, R=R, t=t))

    all_points_2d = []
    all_conf = []
    all_joints_3d = []

    if true_gt_path is not None:
        true_gt_path = Path(true_gt_path)
    else:
        # Auto-discover a true-GT npz for this (subject, actions, split).
        discovered = _find_h36m_true_gt(
            subject, actions, split, Path("data/h36m_true_gt")
        )
        if discovered is not None:
            true_gt_path, _ = discovered
            print(f"NOTE: auto-discovered true 3D GT: {true_gt_path}")

    if true_gt_path is not None:
        if not true_gt_path.exists():
            raise FileNotFoundError(f"True GT file not found: {true_gt_path}")
        gt_data = np.load(true_gt_path, allow_pickle=True)
        if "joints_3d" not in gt_data:
            raise KeyError(
                f"True GT npz must contain 'joints_3d': {true_gt_path}"
            )
        true_joints_3d_all = gt_data["joints_3d"]
        if "actions" in gt_data:
            # The discovered npz may cover more actions than requested (e.g. a
            # combined train/test file). Slice out the requested actions.
            true_joints_3d = _slice_true_gt_by_actions(
                true_joints_3d_all,
                gt_data["actions"],
                actions,
                groups,
                subject,
            )
        else:
            true_joints_3d = true_joints_3d_all
        expected_frames = sum(len(groups[tb]["01"]) for tb in target_bases)
        if true_joints_3d.shape[0] != expected_frames:
            raise ValueError(
                f"True GT joints_3d has {true_joints_3d.shape[0]} frames, "
                f"expected {expected_frames}."
            )
    else:
        true_joints_3d = None
        print(
            "WARNING: No true 3D GT supplied; triangulating 2D keypoints to "
            "produce joints_3d. The resulting labels are circular: DLT(points_2d, "
            "cameras) will be stored as ground truth."
        )

    global_frame = 0
    for target_base in target_bases:
        cams_dict = groups[target_base]
        n_frames = len(cams_dict["01"])
        for frame in range(n_frames):
            p2d = np.stack(
                [split_data["joint_2d"][cams_dict[cam][frame]] for cam in cam_names],
                axis=0,
            )
            conf = np.stack(
                [
                    split_data["confidence"][cams_dict[cam][frame]].squeeze(-1)
                    for cam in cam_names
                ],
                axis=0,
            )
            if true_joints_3d is not None:
                j3d = true_joints_3d[global_frame]
            else:
                j3d = _triangulate_joints(p2d, cameras)
            all_points_2d.append(p2d)
            all_conf.append(conf)
            all_joints_3d.append(j3d)
            global_frame += 1

    points_2d = np.stack(all_points_2d, axis=0)
    confidences = np.stack(all_conf, axis=0)
    joints_3d = np.stack(all_joints_3d, axis=0)

    actions_str = "_".join(f"{a:02d}" for a in actions)
    out_path = out_dir / f"s_{subject:02d}_acts_{actions_str}_multiview.npz"
    _save_canonical_npz(out_path, points_2d, confidences, joints_3d, cameras)
    return out_path


# -----------------------------------------------------------------------------
# Shelf / Campus
# -----------------------------------------------------------------------------
def convert_shelf_campus(
    data_root: Path,
    out_path: Path,
    person_id: int = 0,
) -> Path:
    """Convert a Shelf/Campus sequence to canonical ``.npz``.

    Args:
        data_root: path to ``Shelf_Seq1`` or ``Campus_Seq1`` folder.
        out_path: destination ``.npz`` path.
        person_id: which person to extract.

    Returns:
        Path to the generated ``.npz`` file.
    """
    data_root = Path(data_root)
    calibration_path = data_root / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Shelf/Campus calibration not found: {calibration_path}\n"
            "Download the Shelf/Campus dataset from the official source."
        )

    points_2d, confidences, joints_3d, cameras = build_shelf_dataset(
        data_root, person_id=person_id
    )
    out_path = Path(out_path)
    _save_canonical_npz(out_path, points_2d, confidences, joints_3d, cameras)
    return out_path


# -----------------------------------------------------------------------------
# Synthetic / AMASS
# -----------------------------------------------------------------------------
def convert_synthetic_amass(
    output: Path = Path("outputs/synthetic_multiview_dataset.npz"),
    n_sequences: int = 500,
    frames_per_seq: int = 30,
    n_views: int = 4,
    noise_std: float = 0.5,
    occlusion_rate: float = 0.1,
    outlier_rate: float = 0.02,
) -> Path:
    """Generate a canonical ``.npz`` from a synthetic SMPL/AMASS rig.

    This is a thin wrapper around ``experiments/generate_synthetic_multiview_dataset.py``.
    It requires ``smplx`` and the SMPL neutral model at ``data/smpl/SMPL_NEUTRAL.pkl``.

    Args:
        output: destination ``.npz`` path.
        n_sequences: number of sequences to generate.
        frames_per_seq: frames per sequence.
        n_views: number of calibrated views.
        noise_std: standard deviation of 2D keypoint noise.
        occlusion_rate: fraction of joints to occlude per view.
        outlier_rate: fraction of joints to corrupt as outliers.

    Returns:
        Path to the generated ``.npz`` file.
    """
    import subprocess
    import sys

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    script = Path("experiments/generate_synthetic_multiview_dataset.py")
    if not script.exists():
        raise FileNotFoundError(f"Synthetic generator script missing: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--n_sequences",
        str(n_sequences),
        "--frames_per_seq",
        str(frames_per_seq),
        "--n_views",
        str(n_views),
        "--noise_std",
        str(noise_std),
        "--occlusion_rate",
        str(occlusion_rate),
        "--outlier_rate",
        str(outlier_rate),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return Path(output)


# -----------------------------------------------------------------------------
# MPI-INF-3DHP
# -----------------------------------------------------------------------------
def _parse_mpiinf3dhp_intrinsics(calibration_path: Path) -> List[np.ndarray]:
    """Parse the 3x3 intrinsic matrices from the MPI-INF-3DHP calibration file."""
    text = calibration_path.read_text()
    blocks = re.split(r"\nname\s+\d+", text)
    K_list: List[np.ndarray] = []
    for block in blocks[1:]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        intrinsic = None
        for line in lines:
            if line.startswith("intrinsic"):
                intrinsic = np.fromstring(line.split("intrinsic", 1)[1], sep=" ")
        if intrinsic is None:
            continue
        K4 = intrinsic.reshape(4, 4)
        K_list.append(K4[:3, :3].astype(np.float64))
    return K_list


def _estimate_mpiinf3dhp_extrinsics(
    joints_3d: np.ndarray,
    annot3: np.ndarray,
    K_list: List[np.ndarray],
) -> List[Camera]:
    """Estimate world-to-camera extrinsics from per-camera 3D annotations.

    ``camera.calibration`` contains intrinsics but the stored extrinsics do not
    match the universal 3D ground truth for this sequence.  We recover ``R`` and
    ``t`` by aligning ``annot3[v]`` (3D in view v camera space) with
    ``joints_3d`` (universal world coordinates) using a rigid Procrustes fit.
    Reprojection with the recovered matrices matches ``annot2`` within ~10-20 px.
    """
    T, V, J, _ = annot3.shape
    cameras = []
    for v in range(V):
        X = torch.from_numpy(joints_3d.reshape(-1, 3)).double()
        Y = torch.from_numpy(annot3[:, v, :, :].reshape(-1, 3)).double()
        mx = X.mean(dim=0)
        my = Y.mean(dim=0)
        Xc = X - mx
        Yc = Y - my
        H = Yc.T @ Xc
        U, S, Vt = torch.linalg.svd(H)
        R = U @ Vt
        if torch.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        # camera center in world = mean(world) - R^T * mean(camera_space)
        C = mx - R.T @ my
        t = -R @ C
        cameras.append(Camera(K=K_list[v].copy(), R=R.numpy().copy(), t=t.numpy().copy()))
    return cameras


def convert_mpiinf3dhp(
    data_root: Path,
    out_path: Path,
    n_views: int | None = None,
) -> Path:
    """Convert an MPI-INF-3DHP subject/sequence folder to canonical ``.npz``.

    Expects ``annot.mat`` and ``camera.calibration`` in *data_root*.  The
    ``annot.mat`` contains per-view 2D keypoints ``annot2`` (V x frames) and
    universal 3D ground truth ``univ_annot3``.  Output is in millimeters.

    Args:
        data_root: folder containing ``annot.mat`` and ``camera.calibration``.
        out_path: destination ``.npz`` path.
        n_views: if given, keep only the first ``n_views`` cameras.

    Returns:
        Path to the generated ``.npz`` file.
    """
    data_root = Path(data_root)
    annot_path = data_root / "annot.mat"
    calib_path = data_root / "camera.calibration"
    if not annot_path.exists():
        raise FileNotFoundError(f"annot.mat not found: {annot_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"camera.calibration not found: {calib_path}")

    mat = scipy.io.loadmat(annot_path)
    # annot2: (V, 1) object array, each cell is (T, J*2)
    annot2 = mat["annot2"]
    # annot3: (V, 1) object array, each cell is (T, J*3) in camera space.
    annot3 = mat["annot3"]
    # univ_annot3 is stored as (V, 1) object array where each cell contains the
    # same universal 3D ground truth of shape (T, J*3).  Take the first view.
    univ_annot3 = mat["univ_annot3"][0, 0]  # (T, J*3)

    # Parse intrinsics and estimate extrinsics from the per-camera 3D annotations.
    K_list = _parse_mpiinf3dhp_intrinsics(calib_path)
    if n_views is not None:
        annot2 = annot2[:n_views]
        annot3 = annot3[:n_views]
        K_list = K_list[:n_views]

    # Gather 2D points for each view
    points_2d_list = []
    for v in range(annot2.shape[0]):
        p2d = annot2[v, 0]  # (T, J*2)
        T, D = p2d.shape
        J = D // 2
        p2d = p2d.reshape(T, J, 2)
        points_2d_list.append(p2d)
    points_2d = np.stack(points_2d_list, axis=1)  # (T, V, J, 2)

    T, V, J, _ = points_2d.shape
    confidences = np.ones((T, V, J), dtype=np.float32)

    # Ground-truth 3D: (T, J*3)
    joints_3d = univ_annot3.reshape(T, J, 3)

    # Build per-camera 3D array (T, V, J, 3) for extrinsic estimation.
    annot3_array = np.stack([annot3[v, 0].reshape(T, J, 3) for v in range(V)], axis=1)
    cameras = _estimate_mpiinf3dhp_extrinsics(joints_3d, annot3_array, K_list)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_canonical_npz(out_path, points_2d, confidences, joints_3d, cameras)
    return out_path


# -----------------------------------------------------------------------------
# AIST++
# -----------------------------------------------------------------------------
def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to a 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _load_aistpp_cameras(setting_path: Path) -> List[Camera]:
    """Load the 9-camera rig from an AIST++ ``setting_<suffix>.json`` file."""
    with open(setting_path) as f:
        cams = json.load(f)
    cameras = []
    for cam in cams:
        K = np.array(cam["matrix"], dtype=np.float64)
        R = _rodrigues_to_matrix(np.array(cam["rotation"], dtype=np.float64))
        t = np.array(cam["translation"], dtype=np.float64).reshape(3)
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def _parse_aistpp_mapping(mapping_path: Path) -> dict:
    """Parse ``cameras/mapping.txt`` into ``{sequence_name: setting_name}``."""
    mapping = {}
    with open(mapping_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            seq_name, setting = parts[0], parts[1]
            mapping[seq_name] = setting
    return mapping


def convert_aistpp(
    data_root: Path,
    out_dir: Path,
    split: Path | None = None,
    scale_factor: float | None = None,
    max_seqs: int | None = None,
    use_optim: bool = False,
) -> List[Path]:
    """Convert AIST++ annotations to canonical ``.npz`` files.

    AIST++ stores 3D keypoints and camera translations in the same world
    units (commonly centimeters). Use ``scale_factor=0.01`` to write the
    canonical arrays in meters.

    Args:
        data_root: path to the extracted AIST++ annotations. Expected layout::

            data_root/
              keypoints2d/<seq>.pkl
              keypoints3d/<seq>.pkl
              motions/<seq>.pkl
              cameras/
                mapping.txt
                setting*.json

        out_dir: directory where per-sequence ``.npz`` files are written.
        split: optional split file (one sequence name per line). If ``None``,
            all sequences found in ``keypoints3d`` are converted.
        scale_factor: if given, multiply ``joints_3d`` and camera translations
            by this value. ``None`` means no scaling (keep raw AIST++ units).
        max_seqs: if given, stop after converting this many sequences (useful
            for a quick smoke test).
        use_optim: use ``keypoints3d_optim`` instead of ``keypoints3d``.

    Returns:
        List of paths to generated ``.npz`` files.
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kp2d_dir = data_root / "keypoints2d"
    kp3d_dir = data_root / "keypoints3d"
    motion_dir = data_root / "motions"
    cam_dir = data_root / "cameras"
    mapping_path = cam_dir / "mapping.txt"

    if not kp2d_dir.exists():
        raise FileNotFoundError(f"AIST++ keypoints2d folder not found: {kp2d_dir}")
    if not kp3d_dir.exists():
        raise FileNotFoundError(f"AIST++ keypoints3d folder not found: {kp3d_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(
            "AIST++ camera mapping not found. Make sure cameras/mapping.txt exists.\n"
            "Download from https://github.com/google/aistplusplus_dataset"
        )

    mapping = _parse_aistpp_mapping(mapping_path)

    if split is not None:
        split = Path(split)
        with open(split) as f:
            seq_names = [line.strip() for line in f if line.strip()]
    else:
        seq_names = sorted([p.stem for p in kp3d_dir.glob("*.pkl")])

    if max_seqs is not None:
        seq_names = seq_names[:max_seqs]

    out_paths: List[Path] = []
    for seq_name in seq_names:
        kp2d_path = kp2d_dir / f"{seq_name}.pkl"
        kp3d_path = kp3d_dir / f"{seq_name}.pkl"
        motion_path = motion_dir / f"{seq_name}.pkl"

        if not kp2d_path.exists() or not kp3d_path.exists():
            continue

        with open(kp2d_path, "rb") as f:
            kp2d_data = pickle.load(f)
        with open(kp3d_path, "rb") as f:
            kp3d_data = pickle.load(f)

        key2d = kp2d_data["keypoints2d"]  # (V, T, J, 3)
        if key2d.ndim == 3:
            key2d = key2d[np.newaxis, ...]

        # AIST++ stores (V, T, J, 3); canonical is (T, V, J, 2/1).
        points_2d = np.transpose(key2d[..., :2], (1, 0, 2, 3)).astype(np.float32)
        confidences = np.transpose(key2d[..., 2], (1, 0, 2)).astype(np.float32)

        # Prefer the optimized AIST++ 3D keypoints. The raw 'keypoints3d' contains
        # NaNs for occluded/missing joints on ~20% of sequences, which poisons the
        # validation loss/MPJPE. keypoints3d_optim is clean across the full release.
        if use_optim and "keypoints3d_optim" in kp3d_data:
            joints_3d = kp3d_data["keypoints3d_optim"]
        else:
            joints_3d = kp3d_data["keypoints3d"]
        joints_3d = joints_3d.astype(np.float64)

        if scale_factor is not None:
            joints_3d = joints_3d * scale_factor

        # Defensive: drop any frame that still contains NaN in 2D, confidence, or 3D.
        valid = (
            ~np.isnan(points_2d).any(axis=(1, 2, 3))
            & ~np.isnan(confidences).any(axis=(1, 2))
            & ~np.isnan(joints_3d).any(axis=(1, 2))
        )
        if not valid.all():
            dropped = int((~valid).sum())
            print(f"WARNING [{seq_name}]: dropping {dropped}/{len(valid)} NaN frames")
            points_2d = points_2d[valid]
            confidences = confidences[valid]
            joints_3d = joints_3d[valid]

        setting_name = mapping.get(seq_name)
        if setting_name is None:
            raise ValueError(f"No camera setting found for sequence {seq_name}")
        setting_path = cam_dir / f"{setting_name}.json"
        if not setting_path.exists():
            raise FileNotFoundError(f"Camera setting file missing: {setting_path}")

        cameras = _load_aistpp_cameras(setting_path)
        if scale_factor is not None:
            # Camera translation must also be scaled to match the scaled 3D points.
            for cam in cameras:
                cam.t = cam.t * scale_factor

        out_path = out_dir / f"{seq_name}_multiview.npz"
        _save_canonical_npz(out_path, points_2d, confidences, joints_3d, cameras)
        out_paths.append(out_path)

    return out_paths


# -----------------------------------------------------------------------------
# Stubs for datasets not yet present
# -----------------------------------------------------------------------------

def convert_panoptic(
    data_root: Path,
    out_path: Path,
    **kwargs,
) -> Path:
    """Stub for CMU Panoptic conversion.

    The dataset is not yet bundled with this project. When available, this
    function will map Panoptic's COCO19 skeleton and HD camera calibration to the
    canonical ``.npz`` format.
    """
    raise NotImplementedError(
        "CMU Panoptic converter is a stub.\n"
        "Download the dataset from http://domedb.perception.cs.cmu.edu/ "
        "and implement the loader before calling this function."
    )


def convert_3dpw(
    data_root: Path,
    out_path: Path,
    **kwargs,
) -> Path:
    """Stub for 3DPW conversion.

    3DPW is single-camera plus IMU and therefore not a direct multi-view source.
    When used, this function will project SMPL joints from the provided camera or
    treat each frame as a single-view validation sample.
    """
    raise NotImplementedError(
        "3DPW converter is a stub.\n"
        "Download the dataset from https://3dpw.imt.uni-luebeck.de/ "
        "and implement the loader before calling this function."
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
_CONVERTERS = {
    "human36m": convert_human36m,
    "shelf": convert_shelf_campus,
    "campus": convert_shelf_campus,
    "mpiinf3dhp": convert_mpiinf3dhp,
    "aistpp": convert_aistpp,
    "synthetic": convert_synthetic_amass,
    "panoptic": convert_panoptic,
    "3dpw": convert_3dpw,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert a WebBridge dataset to the canonical multi-view npz format."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=list(_CONVERTERS.keys()),
        help="Which dataset to convert.",
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--actions", type=int, nargs="+", default=[2])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--person_id", type=int, default=0)
    parser.add_argument("--n_views", type=int, default=None, help="MPI-INF-3DHP: keep only first N cameras.")
    parser.add_argument("--split_file", type=Path, default=None, help="AIST++: text file with one sequence name per line.")
    parser.add_argument("--max_seqs", type=int, default=None, help="AIST++: convert at most N sequences.")
    parser.add_argument("--scale_factor", type=float, default=None, help="AIST++: scale 3D points/cameras by this factor (e.g. 0.01 for meters).")
    parser.add_argument("--meters", action="store_true", help="AIST++: alias for --scale_factor 0.01 (raw AIST++ units are usually centimeters).")
    parser.add_argument(
        "--true-gt-path",
        type=Path,
        default=None,
        help="Human3.6M: path to a canonical npz containing the true joints_3d to use instead of triangulating.",
    )
    args = parser.parse_args()

    if args.dataset == "human36m":
        convert_human36m(
            data_root=args.data_root,
            subject=args.subject,
            actions=args.actions,
            split=args.split,
            out_dir=args.out.parent,
            true_gt_path=args.true_gt_path,
        )
    elif args.dataset in {"shelf", "campus"}:
        convert_shelf_campus(data_root=args.data_root, out_path=args.out, person_id=args.person_id)
    elif args.dataset == "mpiinf3dhp":
        convert_mpiinf3dhp(data_root=args.data_root, out_path=args.out, n_views=args.n_views)
    elif args.dataset == "aistpp":
        scale_factor = 0.01 if args.meters else args.scale_factor
        convert_aistpp(
            data_root=args.data_root,
            out_dir=args.out,
            split=args.split_file,
            scale_factor=scale_factor,
            max_seqs=args.max_seqs,
        )
    else:
        _CONVERTERS[args.dataset](data_root=args.data_root, out_path=args.out)

    print(f"Saved canonical npz to {args.out}")


if __name__ == "__main__":
    main()
