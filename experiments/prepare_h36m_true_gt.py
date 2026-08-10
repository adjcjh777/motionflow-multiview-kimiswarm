"""Extract true 3D ground truth from the official Human3.6M mocap release.

The project's H36M ``.npz`` labels are circular (``joints_3d`` is the DLT
triangulation of the input 2D; see ``scripts/diagnose_circular_labels.py``).
To repair the data foundation, the original mocap world coordinates
(``PosesD3_Positions``) must be used as ``joints_3d`` instead.

This script reads the official per-subject mocap files (``.cdf`` from the
Human3.6M release, or ``.mat`` converted with the common cdf->mat tools),
aligns them frame-by-frame with the frame order produced by
``experiments/prepare_h36m_multiview.py`` / ``convert_human36m`` for a given
subject/action/split of ``h36m_sh_conf_cam_source_final.pkl``, and writes an
``.npz`` with a ``joints_3d`` array that can be passed straight back in via
``--true-gt-path``:

    python experiments/prepare_h36m_true_gt.py \
        --mocap_dir data/h36m_true_gt --subject 9 --actions 2 \
        --split test --out data/h36m_true_gt/s_09_act_02_true_gt.npz

    python experiments/prepare_h36m_multiview.py \
        --subject 9 --actions 2 --split test \
        --true-gt-path data/h36m_true_gt/s_09_act_02_true_gt.npz

Expected mocap layout (either naming style works)::

    data/h36m_true_gt/
        S9/
            Directions 1.54138969.mat      (or .cdf)
            Directions 2.60457474.mat
            ...

Units: the official mocap is in millimetres, matching the canonical H36M
``.npz`` convention (camera_t norm ~5e3).  Pass ``--meters`` to write metres.

Joint convention: 17 joints are selected from the 32 mocap joints.  The
default selection is the first 17 named joints of the official skeleton
(Hip ... Right Wrist, README order).  If the reprojection audit
(``scripts/check_true_gt_reprojection.py``) reports large errors once the
labels are wired in, pass an explicit ``--joint-indices`` list matching the
17-joint order used by the 2D detections in the pkl.

VideoPose3D-format source (``--data3d-npz``)
--------------------------------------------

The MHFormer/VideoPose3D public release ``data_3d_h36m.npz`` (Google Drive
file id ``1mAHq0YhO75frDkgUgebFQYnnPQOjUcr4``; ~174 MB) stores the same
mocap-derived 3D joints as a pickled dict::

    positions_3d -> {"S1": {"Directions 1": (F, 32, 3) float32, ...}, ...}

Units are METRES (root height ~0.98), unlike the official release which is in
millimetres; this source path scales by 1000 to match the project's mm camera
convention.  All 32 mocap joints are kept, but the 17-joint subset that
matches the pkl's 2D detections is NOT the first 17; it is the standard
H36M-17 skeleton ``[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]`` (verified:
reprojection RMSE ~3-8 px vs ~260 px for ``range(17)``).

IMPORTANT action-id convention: the pkl action ids follow the
karfly/human36m-camera-parameters order, which differs from the official
release order used by ``ACTION_NAMES`` above for ids 7 and 12-16.  The pkl's
own ``action`` strings and reprojection give::

    2 Directions, 3 Discussion, 4 Eating, 5 Greeting, 6 Phoning, 7 Posing,
    8 Purchases, 9 Sitting, 10 SittingDown, 11 Smoking, 12 Photo,
    13 Waiting, 14 Walking, 15 WalkDog, 16 WalkTogether

Usage::

    python experiments/prepare_h36m_true_gt.py \
        --data3d-npz data/h36m_true_gt/data_3d_h36m.npz \
        --subject 9 --actions 2 14 --split test \
        --out data/h36m_true_gt/s_09_acts_02_14_true_gt.npz

Author: research swarm (data foundation repair, 2026-08)
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).parent.parent))

# Standard H36M action-id -> action-name mapping (ids as used in the
# ``s_XX_act_YY`` pkl source strings).
ACTION_NAMES: Dict[int, str] = {
    2: "Directions",
    3: "Discussion",
    4: "Eating",
    5: "Greeting",
    6: "Phoning",
    7: "Photo",
    8: "Purchases",
    9: "Sitting",
    10: "SittingDown",
    11: "Smoking",
    12: "Waiting",
    13: "WalkDog",
    14: "Walking",
    15: "WalkTogether",
}

N_MOCAP_JOINTS = 32

# Action-id -> name mapping as used by the pkl's ``action`` strings /
# karfly human36m-camera-parameters convention.  This is the ordering the
# VideoPose3D-format ``data_3d_h36m.npz`` must be sliced with; note it differs
# from the official release ordering (ACTION_NAMES) for ids 7 and 12-16.
ACTION_NAMES_PKL: Dict[int, str] = {
    2: "Directions",
    3: "Discussion",
    4: "Eating",
    5: "Greeting",
    6: "Phoning",
    7: "Posing",
    8: "Purchases",
    9: "Sitting",
    10: "SittingDown",
    11: "Smoking",
    12: "Photo",
    13: "Waiting",
    14: "Walking",
    15: "WalkDog",
    16: "WalkTogether",
}

# Standard H36M 17-joint subset of the 32 mocap joints, in the order used by
# the pkl's 2D detections (and by MHFormer/VideoPose3D downstream).
VP3D_JOINT_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27)

# VideoPose3D-format npz stores metres; project convention is millimetres.
VP3D_METERS_TO_MM: float = 1000.0

# First 17 named joints of the official 32-joint skeleton (README order):
# Hip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine, Thorax, Neck/Head,
# Head Top, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist.
DEFAULT_JOINT_INDICES: Tuple[int, ...] = tuple(range(17))


def load_mocap_frames(path: Path) -> np.ndarray:
    """Load one mocap file and return ``(F, 32, 3)`` positions in mm.

    Supported formats:

    * ``.mat`` produced by the common cdf->mat conversion tools.  The pose
      array is looked up under ``Pose`` (cell arrays from the MATLAB
      conversion are unwrapped), ``data``, or the first non-underscore key.
      Arrays of shape ``(96, F)`` or ``(F, 96)`` are reshaped to ``(F, 32, 3)``
      with joint-major layout ``[x1, y1, z1, x2, ...]``.
    * ``.cdf`` from the official release (requires ``spacepy``).
    """
    path = Path(path)
    if path.suffix.lower() == ".mat":
        mat = scipy.io.loadmat(path)
        arr = None
        for key in ("Pose", "data"):
            if key in mat:
                arr = mat[key]
                break
        if arr is None:
            for key, value in mat.items():
                if not key.startswith("__"):
                    arr = value
                    break
        if arr is None:
            raise ValueError(f"No pose array found in {path}")
        # Unwrap MATLAB cell arrays produced by cdf->mat conversion.
        while arr.dtype == object:
            arr = arr.reshape(-1)[0]
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3:  # already (F, 32, 3) or (32, 3, F)-like
            if arr.shape[1:] == (N_MOCAP_JOINTS, 3):
                return arr
            if arr.shape[0:1] == (N_MOCAP_JOINTS,) and arr.shape[-1] != 3:
                arr = np.transpose(arr, (2, 0, 1))
                if arr.shape[1:] == (N_MOCAP_JOINTS, 3):
                    return arr
            raise ValueError(f"Unsupported mocap shape {arr.shape} in {path}")
        if arr.ndim != 2 or N_MOCAP_JOINTS * 3 not in arr.shape:
            raise ValueError(f"Unsupported mocap shape {arr.shape} in {path}")
        if arr.shape[0] == N_MOCAP_JOINTS * 3:
            arr = arr.T  # (96, F) -> (F, 96)
        return arr.reshape(arr.shape[0], N_MOCAP_JOINTS, 3)

    if path.suffix.lower() == ".cdf":
        try:
            from spacepy import pycdf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"Reading {path} requires the 'spacepy' package "
                "(pip install spacepy), or convert the official .cdf files "
                "to .mat first (e.g. with the H36M cdf->mat MATLAB tools)."
            ) from exc
        with pycdf.CDF(str(path)) as cdf:
            if "Pose" not in cdf:
                raise ValueError(f"No 'Pose' variable in {path}")
            arr = np.asarray(cdf["Pose"][...], dtype=np.float64)
        if arr.shape[0] == N_MOCAP_JOINTS * 3:
            arr = arr.T
        return arr.reshape(arr.shape[0], N_MOCAP_JOINTS, 3)

    raise ValueError(f"Unsupported mocap file type: {path}")


def resolve_mocap_files(
    mocap_dir: Path,
    subject: int,
    action_id: int,
    action_name: Optional[str] = None,
) -> Dict[int, Path]:
    """Map subaction id (1, 2, ...) to mocap file paths for one subject/action.

    Matches file names like ``Directions 1.54138969.mat`` or
    ``Directions 1.cdf`` inside ``mocap_dir/S{subject}/``.
    """
    mocap_dir = Path(mocap_dir)
    name = action_name or ACTION_NAMES.get(action_id)
    if name is None:
        raise ValueError(
            f"No known action name for action id {action_id}. "
            f"Known ids: {sorted(ACTION_NAMES)}. Pass --action-name explicitly."
        )
    subj_dir = mocap_dir / f"S{subject}"
    if not subj_dir.is_dir():
        raise FileNotFoundError(
            f"Mocap subject directory not found: {subj_dir}\n"
            "Download the official Human3.6M release (Subjects_*.tgz, "
            "MyPoseFeatures/D3_Positions) and extract it as "
            "<mocap_dir>/S<subject>/<Action> <sub>.{cdf,mat}."
        )
    found: Dict[int, Path] = {}
    for candidate in sorted(subj_dir.iterdir()):
        m = re.match(rf"^{re.escape(name)}\s+(\d+)", candidate.name)
        if m is None:
            continue
        found[int(m.group(1))] = candidate
    if not found:
        available = sorted(p.name for p in subj_dir.iterdir())[:20]
        raise FileNotFoundError(
            f"No mocap files for action '{name}' in {subj_dir}. "
            f"Available files (first 20): {available}. "
            "If the naming differs, pass --action-name."
        )
    return found


def pkl_frame_counts(
    pkl_path: Path,
    split: str,
    subject: int,
    actions: Sequence[int],
) -> List[Tuple[str, int, Optional[int]]]:
    """Return ``(base_source, n_frames, subaction)`` groups in converter order.

    The grouping mirrors ``convert_human36m`` / ``prepare_h36m_multiview.py``:
    a source base is everything before the trailing ``_cam_NN`` / ``_ca_NN``
    token (so test-split bases keep their ``_subact_NN`` suffix), and bases
    are processed per action in sorted order.  ``n_frames`` is the number of
    frames per camera for that group, i.e. exactly the number of
    ``joints_3d`` frames the canonical converter emits for it.
    """
    pkl_path = Path(pkl_path)
    if pkl_path.suffix == ".zip":
        with zipfile.ZipFile(pkl_path) as z:
            inner = [n for n in z.namelist() if n.endswith(".pkl")]
            if not inner:
                raise ValueError(f"No .pkl inside {pkl_path}")
            with z.open(inner[0]) as f:
                data = pickle.load(f)
    else:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

    split_data = data[split]
    groups: Dict[str, Dict[str, List[int]]] = {}
    for i, src in enumerate(split_data["source"]):
        m = re.match(r"(.+)_(?:cam|ca)_(\d+)$", src)
        if m is None:
            raise ValueError(f"Unsupported source format: {src}")
        base, cam = m.group(1), m.group(2)
        groups.setdefault(base, {}).setdefault(cam, []).append(i)

    prefix = f"s_{subject:02d}_act_"
    out: List[Tuple[str, int, Optional[int]]] = []
    for action in actions:
        action_prefix = f"s_{subject:02d}_act_{action:02d}"
        matches = sorted(
            b
            for b in groups.keys()
            if b.startswith(action_prefix) and b.startswith(prefix)
        )
        if not matches:
            raise ValueError(
                f"No source found for subject {subject} action {action} in {split}"
            )
        for base in matches:
            first_cam = sorted(groups[base].keys())[0]
            sub_m = re.search(r"subact_(\d+)$", base)
            subaction = int(sub_m.group(1)) if sub_m else None
            out.append((base, len(groups[base][first_cam]), subaction))
    return out


def build_true_gt(
    mocap_dir: Path,
    pkl_path: Path,
    split: str,
    subject: int,
    actions: Sequence[int],
    joint_indices: Sequence[int] = DEFAULT_JOINT_INDICES,
    action_name: Optional[str] = None,
) -> np.ndarray:
    """Assemble ``(T, len(joint_indices), 3)`` true 3D GT in pkl frame order."""
    groups = pkl_frame_counts(pkl_path, split, subject, actions)
    chunks: List[np.ndarray] = []
    for base, n_frames, subaction in groups:
        m = re.match(rf"s_{subject:02d}_act_(\d+)", base)
        if m is None:
            raise ValueError(f"Cannot parse action id from base {base}")
        action_id = int(m.group(1))
        files = resolve_mocap_files(mocap_dir, subject, action_id, action_name)
        if subaction is not None:
            if subaction not in files:
                raise FileNotFoundError(
                    f"No mocap file for {base} (available subactions: {sorted(files)})"
                )
            sub_ids = [subaction]
        else:
            sub_ids = sorted(files.keys())
        frames = [load_mocap_frames(files[sub]) for sub in sub_ids]
        mocap_all = np.concatenate(frames, axis=0)
        if mocap_all.shape[0] < n_frames:
            raise ValueError(
                f"Mocap for {base} has {mocap_all.shape[0]} frames but the pkl "
                f"expects {n_frames}; cannot align. Check that the official "
                "mocap release matches the pkl's H36M version."
            )
        if mocap_all.shape[0] > n_frames:
            print(
                f"  {base}: truncating mocap {mocap_all.shape[0]} -> {n_frames} frames"
            )
        chunks.append(mocap_all[:n_frames])
    joints_32 = np.concatenate(chunks, axis=0)
    return joints_32[:, list(joint_indices), :]


def build_true_gt_from_data3d_npz(
    data3d_npz: Path,
    pkl_path: Path,
    split: str,
    subject: int,
    actions: Sequence[int],
    joint_indices: Sequence[int] = VP3D_JOINT_INDICES,
    action_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Assemble ``(T, len(joint_indices), 3)`` true 3D GT in mm from the
    VideoPose3D-format ``data_3d_h36m.npz`` in pkl frame order.

    The npz holds a single pickled array ``positions_3d`` mapping
    ``"S{subject}"`` -> ``{"<Action> [sub]": (F, 32, 3) float32}`` with all 32
    mocap joints in metres.  Values are multiplied by 1000 so the output is in
    millimetres, matching the official-release pipeline and the camera
    parameters.  Unlike the per-action files of the official release, this
    source stores every subaction as a separate key
    (``"Directions"``, ``"Directions 1"``, ...).

    For each pkl source group the matching key is chosen by exact frame count
    and confirmed by a reprojection score; when several keys share the exact
    frame count (identical subactions, e.g. Greeting/SittingDown), the
    lowest-reprojection-error key is used.  Groups are concatenated in the
    same order ``convert_human36m`` / ``prepare_h36m_multiview.py`` emit them
    (actions in the order given, bases sorted within each action).
    """
    import json

    names = ACTION_NAMES_PKL if action_names is None else action_names
    data3d_npz = Path(data3d_npz)
    arr = np.load(data3d_npz, allow_pickle=True)["positions_3d"]
    positions = arr.item() if arr.ndim == 0 else arr
    subj_key = f"S{subject}"
    if subj_key not in positions:
        raise KeyError(
            f"{data3d_npz} has no subject {subj_key}; available: {sorted(positions)}"
        )
    subj = positions[subj_key]

    groups = pkl_frame_counts(pkl_path, split, subject, actions)

    # Preload pkl 2D + cameras once for confirmation reprojection.
    pkl_path = Path(pkl_path)
    if pkl_path.suffix == ".zip":
        with zipfile.ZipFile(pkl_path) as z:
            inner = [n for n in z.namelist() if n.endswith(".pkl")]
            with z.open(inner[0]) as f:
                pkl = pickle.load(f)
    else:
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
    split_data = pkl[split]

    cam_params = None
    cam_params_path = Path("data/h36m_hf/camera_params.json")
    if cam_params_path.exists():
        with open(cam_params_path) as f:
            cam_params = json.load(f)

    import re as _re

    # Index pkl rows by (base, camera-slot) for reprojection confirmation.
    rows_by_base: Dict[str, Dict[str, List[int]]] = {}
    for i, src in enumerate(split_data["source"]):
        m = _re.match(r"(.+)_(?:cam|ca)_(\d+)$", src)
        if m is None:
            continue
        rows_by_base.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(i)

    SLOT_CAMERAS = {"01": "54138969", "02": "55011271", "03": "58860488", "04": "60457274"}

    def reproj_rmse_frames(X: np.ndarray, base: str, n_frames: int) -> Optional[float]:
        """RMSE (px) of reprojecting world frames ``X`` (n_frames, J, 3, mm)."""
        if cam_params is None:
            return None
        if "joint_2d" not in split_data or "confidence" not in split_data:
            return None
        cams = rows_by_base.get(base)
        if not cams:
            return None
        errs = []
        for slot in sorted(cams):
            camera_name = SLOT_CAMERAS.get(slot)
            if camera_name is None or camera_name not in cam_params["intrinsics"]:
                continue
            rows = cams[slot][:n_frames]
            if len(rows) != n_frames:
                continue
            K = np.array(cam_params["intrinsics"][camera_name]["calibration_matrix"], dtype=np.float64)
            ext = cam_params["extrinsics"][subj_key][camera_name]
            R = np.array(ext["R"], dtype=np.float64)
            t = np.array(ext["t"], dtype=np.float64).reshape(3)
            Xc = np.einsum("ij,nfj->nfi", R, X) + t
            uv = np.stack(
                [K[0, 0] * Xc[..., 0] / Xc[..., 2] + K[0, 2],
                 K[1, 1] * Xc[..., 1] / Xc[..., 2] + K[1, 2]],
                axis=-1,
            )
            j2d = np.stack([split_data["joint_2d"][r] for r in rows])
            conf = np.stack([split_data["confidence"][r].squeeze(-1) for r in rows])
            mask = conf > 0.5
            e = np.sqrt(((uv - j2d) ** 2).sum(-1))[mask]
            errs.append(e)
        if not errs:
            return None
        all_e = np.concatenate(errs)
        return float(np.sqrt((all_e ** 2).mean()))

    def reproj_rmse(key: str, base: str, n_frames: int) -> Optional[float]:
        X = np.asarray(subj[key], dtype=np.float64)[:n_frames] * VP3D_METERS_TO_MM
        return reproj_rmse_frames(X[:, list(joint_indices), :], base, n_frames)

    chunks: List[np.ndarray] = []
    for base, n_frames, _subaction in groups:
        m = _re.match(rf"s_{subject:02d}_act_(\d+)", base)
        if m is None:
            raise ValueError(f"Cannot parse action id from base {base}")
        action_id = int(m.group(1))
        name = names.get(action_id)
        if name is None:
            raise ValueError(f"No known action name for id {action_id}")
        candidates = [k for k in subj if k == name or k.startswith(name + " ")]
        if not candidates:
            raise KeyError(
                f"{data3d_npz}: no key starting with {name!r} for S{subject}; "
                f"available: {sorted(subj)}"
            )
        exact = [k for k in candidates if np.asarray(subj[k]).shape[0] == n_frames]
        chosen: Optional[np.ndarray] = None
        if len(exact) == 1:
            key = exact[0]
            score = reproj_rmse(key, base, n_frames)
            if score is not None and score > 20.0:
                print(
                    f"  WARNING {base}: exact-frame-count key {key!r} reprojection "
                    f"RMSE {score:.1f} px (>20); frame alignment is suspect."
                )
            chosen = np.asarray(subj[key], dtype=np.float64)[:n_frames] * VP3D_METERS_TO_MM
        elif exact:
            scored = [(reproj_rmse(k, base, n_frames), k) for k in exact]
            scored = [(s, k) for s, k in scored if s is not None]
            if not scored:
                raise RuntimeError(
                    f"Multiple exact-frame-count keys {exact} for {base} and "
                    "no reprojection data to disambiguate."
                )
            score, key = min(scored)
            if score > 20.0:
                print(
                    f"  WARNING {base}: best of {exact} is {key!r} with RMSE "
                    f"{score:.1f} px; check that this action exists in the pkl."
                )
            else:
                print(f"  {base}: disambiguated {exact} -> {key!r} (RMSE {score:.1f} px)")
            chosen = np.asarray(subj[key], dtype=np.float64)[:n_frames] * VP3D_METERS_TO_MM
        else:
            # Train-split style: one source base per action whose frames are the
            # concatenation of all available subactions.  Search over key
            # orderings (all permutations when few keys) for the one whose
            # concatenation best reprojects to the pkl 2D.
            import itertools as _itertools

            best: Optional[Tuple[float, List[str], np.ndarray]] = None
            for perm in _itertools.permutations(candidates):
                total = sum(int(np.asarray(subj[k]).shape[0]) for k in perm)
                if total < n_frames:
                    continue
                cat = np.concatenate([np.asarray(subj[k], dtype=np.float64) for k in perm], axis=0)
                cat = cat[:n_frames] * VP3D_METERS_TO_MM
                score = reproj_rmse_frames(cat[:, list(joint_indices), :], base, n_frames)
                if score is None:
                    if best is None:
                        best = (float("inf"), list(perm), cat)
                    continue
                if best is None or score < best[0]:
                    best = (score, list(perm), cat)
                    if score <= 20.0:
                        break  # good enough
            if best is None:
                sizes = {k: int(np.asarray(subj[k]).shape[0]) for k in candidates}
                raise ValueError(
                    f"{base}: no ordering of {name!r} mocap keys covers {n_frames} "
                    f"frames; candidate sizes: {sizes}"
                )
            score, perm_keys, chosen = best
            if score == float("inf"):
                print(
                    f"  WARNING {base}: no reprojection data; using ordering "
                    f"{perm_keys} (unverified)."
                )
            elif score > 20.0:
                print(
                    f"  WARNING {base}: best ordering {perm_keys} reprojects with "
                    f"RMSE {score:.1f} px (>20); frame alignment is suspect."
                )
            else:
                print(f"  {base}: aligned via ordering {perm_keys} (RMSE {score:.1f} px)")
        chunks.append(chosen[:, list(joint_indices), :])
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build true 3D GT joints_3d for H36M from the official mocap release."
    )
    parser.add_argument("--mocap_dir", type=Path, default=Path("data/h36m_true_gt"))
    parser.add_argument(
        "--pkl",
        type=Path,
        default=Path("data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip"),
    )
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--actions", type=int, nargs="+", required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--action-name",
        type=str,
        default=None,
        help="Override the action-id -> file-name mapping (e.g. 'Directions').",
    )
    parser.add_argument(
        "--joint-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Indices into the 32 mocap joints for the 17-joint skeleton. "
            "Default: range(17) for the official mocap dir, and the standard "
            "H36M-17 subset " + str(list(VP3D_JOINT_INDICES)) + " for --data3d-npz."
        ),
    )
    parser.add_argument(
        "--data3d-npz",
        type=Path,
        default=None,
        help=(
            "VideoPose3D-format data_3d_h36m.npz (MHFormer Google Drive release). "
            "When given, mocap files are read from this npz instead of --mocap_dir."
        ),
    )
    parser.add_argument(
        "--meters",
        action="store_true",
        help="Write metres instead of the default millimetres.",
    )
    args = parser.parse_args()

    if args.data3d_npz is not None:
        joint_indices = (
            args.joint_indices if args.joint_indices is not None else list(VP3D_JOINT_INDICES)
        )
        joints_3d = build_true_gt_from_data3d_npz(
            data3d_npz=args.data3d_npz,
            pkl_path=args.pkl,
            split=args.split,
            subject=args.subject,
            actions=args.actions,
            joint_indices=joint_indices,
        )
    else:
        joint_indices = (
            args.joint_indices if args.joint_indices is not None else list(DEFAULT_JOINT_INDICES)
        )
        joints_3d = build_true_gt(
            mocap_dir=args.mocap_dir,
            pkl_path=args.pkl,
            split=args.split,
            subject=args.subject,
            actions=args.actions,
            joint_indices=joint_indices,
            action_name=args.action_name,
        )
    unit = "mm"
    if args.meters:
        joints_3d = joints_3d / 1000.0
        unit = "m"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        joints_3d=joints_3d,
        subject=np.int64(args.subject),
        actions=np.asarray(args.actions, dtype=np.int64),
        joint_indices=np.asarray(joint_indices, dtype=np.int64),
        unit=unit,
    )
    print(
        f"Saved true GT {args.out}: joints_3d {joints_3d.shape} ({unit}), "
        f"range [{joints_3d.min():.1f}, {joints_3d.max():.1f}]"
    )
    print(
        "Next: regenerate the canonical npz with --true-gt-path and audit it "
        "with scripts/check_true_gt_reprojection.py and "
        "scripts/diagnose_circular_labels.py (direct MJE must be >> 0 mm)."
    )


if __name__ == "__main__":
    main()
