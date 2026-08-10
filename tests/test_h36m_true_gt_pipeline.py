"""Tests for the H36M true-GT pipeline (data foundation repair).

Covers:

* ``experiments/prepare_h36m_true_gt.py``: mocap loading (.mat layouts),
  action/subaction file resolution, pkl frame-count grouping, and the
  end-to-end ``build_true_gt`` assembly including frame alignment errors.
* ``scripts/check_true_gt_reprojection.py``: reprojection RMSE audit that
  validates true 3D labels against the stored cameras/2D.
"""

import pickle

import numpy as np
import pytest
import scipy.io

from experiments.prepare_h36m_true_gt import (
    DEFAULT_JOINT_INDICES,
    VP3D_JOINT_INDICES,
    VP3D_METERS_TO_MM,
    build_true_gt,
    build_true_gt_from_data3d_npz,
    load_mocap_frames,
    pkl_frame_counts,
    resolve_mocap_files,
)
from scripts.check_true_gt_reprojection import check_npz, reproject

N_JOINTS = 32


def _mocap_array(n_frames, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_frames, N_JOINTS, 3)).astype(np.float64)


def _write_mat(path, arr3d, key="Pose", layout="F96"):
    flat = arr3d.reshape(arr3d.shape[0], -1)
    if layout == "F96":
        payload = flat
    elif layout == "96F":
        payload = flat.T
    else:
        raise ValueError(layout)
    scipy.io.savemat(path, {key: payload})


# ---------------------------------------------------------------------------
# load_mocap_frames
# ---------------------------------------------------------------------------
def test_load_mocap_frames_F96(tmp_path):
    arr = _mocap_array(7)
    path = tmp_path / "Directions 1.54138969.mat"
    _write_mat(path, arr, key="Pose", layout="F96")
    out = load_mocap_frames(path)
    assert out.shape == (7, N_JOINTS, 3)
    np.testing.assert_allclose(out, arr)


def test_load_mocap_frames_96F(tmp_path):
    arr = _mocap_array(5)
    path = tmp_path / "Walking 1.mat"
    _write_mat(path, arr, key="data", layout="96F")
    out = load_mocap_frames(path)
    assert out.shape == (5, N_JOINTS, 3)
    np.testing.assert_allclose(out, arr)


def test_load_mocap_frames_cell_wrapped(tmp_path):
    arr = _mocap_array(4)
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = arr.reshape(4, -1)
    path = tmp_path / "Sitting 1.mat"
    scipy.io.savemat(path, {"Pose": cell})
    out = load_mocap_frames(path)
    assert out.shape == (4, N_JOINTS, 3)
    np.testing.assert_allclose(out, arr)


def test_load_mocap_frames_bad_shape(tmp_path):
    path = tmp_path / "bad.mat"
    scipy.io.savemat(path, {"Pose": np.zeros((10, 10))})
    with pytest.raises(ValueError, match="Unsupported mocap shape"):
        load_mocap_frames(path)


# ---------------------------------------------------------------------------
# resolve_mocap_files
# ---------------------------------------------------------------------------
def test_resolve_mocap_files(tmp_path):
    subj = tmp_path / "S9"
    subj.mkdir(parents=True)
    f1 = subj / "Directions 1.54138969.mat"
    f2 = subj / "Directions 2.60457474.mat"
    f1.touch()
    f2.touch()
    (subj / "Eating 1.54138969.mat").touch()
    found = resolve_mocap_files(tmp_path, subject=9, action_id=2)
    assert found == {1: f1, 2: f2}


def test_resolve_mocap_files_missing_subject(tmp_path):
    with pytest.raises(FileNotFoundError, match="subject directory not found"):
        resolve_mocap_files(tmp_path, subject=9, action_id=2)


def test_resolve_mocap_files_unknown_action(tmp_path):
    subj = tmp_path / "S9"
    subj.mkdir(parents=True)
    (subj / "Directions 1.mat").touch()
    with pytest.raises(ValueError, match="No known action name"):
        resolve_mocap_files(tmp_path, subject=9, action_id=99)
    with pytest.raises(FileNotFoundError, match="Available files"):
        resolve_mocap_files(tmp_path, subject=9, action_id=2, action_name="Flying")


# ---------------------------------------------------------------------------
# pkl_frame_counts / build_true_gt
# ---------------------------------------------------------------------------
def _write_pkl(path, split, sources):
    with open(path, "wb") as f:
        pickle.dump({split: {"source": sources}}, f)


def test_pkl_frame_counts_train_and_test(tmp_path):
    pkl = tmp_path / "fake.pkl"
    sources = (
        ["s_01_act_02_cam_01"] * 3
        + ["s_01_act_02_cam_02"] * 3
        + ["s_01_act_03_cam_01"] * 2
        + ["s_01_act_03_cam_02"] * 2
    )
    _write_pkl(pkl, "train", sources)
    groups = pkl_frame_counts(pkl, "train", subject=1, actions=[2, 3])
    assert groups == [("s_01_act_02", 3, None), ("s_01_act_03", 2, None)]


def test_pkl_frame_counts_subaction_test_split(tmp_path):
    pkl = tmp_path / "fake.pkl"
    sources = (
        ["s_09_act_02_subact_01_ca_01"] * 4
        + ["s_09_act_02_subact_01_ca_02"] * 4
        + ["s_09_act_02_subact_02_ca_01"] * 2
        + ["s_09_act_02_subact_02_ca_02"] * 2
    )
    _write_pkl(pkl, "test", sources)
    groups = pkl_frame_counts(pkl, "test", subject=9, actions=[2])
    assert groups == [
        ("s_09_act_02_subact_01", 4, 1),
        ("s_09_act_02_subact_02", 2, 2),
    ]


def _setup_mocap(tmp_path, subject, name, frames_per_sub):
    """Create mocap .mat files; return mocap dir and arrays per subaction."""
    subj_dir = tmp_path / "mocap" / f"S{subject}"
    subj_dir.mkdir(parents=True)
    arrays = []
    for sub, n in frames_per_sub.items():
        arr = _mocap_array(n, seed=subject * 100 + sub)
        _write_mat(subj_dir / f"{name} {sub}.54138969.mat", arr)
        arrays.append(arr)
    return tmp_path / "mocap", arrays


def test_build_true_gt_train_concatenates_subactions(tmp_path):
    pkl = tmp_path / "fake.pkl"
    _write_pkl(pkl, "train", ["s_01_act_02_cam_01"] * 10 + ["s_01_act_02_cam_02"] * 10)
    mocap_dir, arrays = _setup_mocap(tmp_path, 1, "Directions", {1: 6, 2: 6})

    gt = build_true_gt(mocap_dir, pkl, "train", subject=1, actions=[2])
    expected = np.concatenate(arrays, axis=0)[:10][:, list(DEFAULT_JOINT_INDICES), :]
    assert gt.shape == (10, 17, 3)
    np.testing.assert_allclose(gt, expected)


def test_build_true_gt_test_split_subactions(tmp_path):
    pkl = tmp_path / "fake.pkl"
    sources = (
        ["s_09_act_02_subact_01_ca_01"] * 4
        + ["s_09_act_02_subact_02_ca_01"] * 3
    )
    _write_pkl(pkl, "test", sources)
    mocap_dir, arrays = _setup_mocap(tmp_path, 9, "Directions", {1: 5, 2: 3})

    gt = build_true_gt(mocap_dir, pkl, "test", subject=9, actions=[2])
    expected = np.concatenate(
        [arrays[0][:4], arrays[1][:3]], axis=0
    )[:, list(DEFAULT_JOINT_INDICES), :]
    assert gt.shape == (7, 17, 3)
    np.testing.assert_allclose(gt, expected)


def test_build_true_gt_too_short_raises(tmp_path):
    pkl = tmp_path / "fake.pkl"
    _write_pkl(pkl, "train", ["s_01_act_02_cam_01"] * 10)
    mocap_dir, _ = _setup_mocap(tmp_path, 1, "Directions", {1: 4})
    with pytest.raises(ValueError, match="cannot align"):
        build_true_gt(mocap_dir, pkl, "train", subject=1, actions=[2])


# ---------------------------------------------------------------------------
# build_true_gt_from_data3d_npz (VideoPose3D-format source)
# ---------------------------------------------------------------------------
def _write_data3d_npz(path, subjects):
    """Write a VideoPose3D-format npz: positions_3d -> 0-d object dict."""
    arr = np.empty((), dtype=object)
    arr[()] = subjects
    np.savez_compressed(path, positions_3d=arr)


def test_data3d_test_split_exact_keys_and_units(tmp_path):
    """Test-split bases pick the exact-frame-count key; metres -> mm scaling."""
    pkl = tmp_path / "fake.pkl"
    _write_pkl(
        pkl,
        "test",
        ["s_09_act_02_subact_01_ca_01"] * 4
        + ["s_09_act_02_subact_02_ca_01"] * 3,
    )
    rng = np.random.default_rng(7)
    # metres (the VideoPose3D convention); build_true_gt_from_data3d_npz must
    # scale by VP3D_METERS_TO_MM.
    d1 = rng.normal(size=(4, N_JOINTS, 3))
    d2 = rng.normal(size=(3, N_JOINTS, 3))
    _write_data3d_npz(
        tmp_path / "data_3d_h36m.npz",
        {"S9": {"Directions 1": d1.astype(np.float32), "Directions": d2.astype(np.float32)}},
    )
    gt = build_true_gt_from_data3d_npz(
        tmp_path / "data_3d_h36m.npz", pkl, "test", subject=9, actions=[2]
    )
    assert gt.shape == (7, len(VP3D_JOINT_INDICES), 3)
    idx = list(VP3D_JOINT_INDICES)
    expected = np.concatenate(
        [d1[:, idx, :], d2[:, idx, :]], axis=0
    ).astype(np.float32) * VP3D_METERS_TO_MM
    np.testing.assert_allclose(gt, expected, rtol=1e-6, atol=1e-3)


def test_data3d_train_split_concatenates_all_orderings(tmp_path):
    """Train bases (sum of subactions) align via best concatenation ordering."""
    pkl = tmp_path / "fake.pkl"
    _write_pkl(pkl, "train", ["s_01_act_02_cam_01"] * 10)
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, N_JOINTS, 3))
    b = rng.normal(size=(4, N_JOINTS, 3))
    _write_data3d_npz(
        tmp_path / "data_3d_h36m.npz",
        {"S1": {"Directions": a.astype(np.float32), "Directions 1": b.astype(np.float32)}},
    )
    gt = build_true_gt_from_data3d_npz(
        tmp_path / "data_3d_h36m.npz", pkl, "train", subject=1, actions=[2]
    )
    assert gt.shape == (10, len(VP3D_JOINT_INDICES), 3)
    idx = list(VP3D_JOINT_INDICES)
    # Without camera data there is no reprojection signal, so the first
    # permutation order is used: the result must match one of the two orders.
    order_ab = np.concatenate([a[:, idx], b[:, idx]], 0).astype(np.float32) * VP3D_METERS_TO_MM
    order_ba = np.concatenate([b[:, idx], a[:, idx]], 0).astype(np.float32) * VP3D_METERS_TO_MM
    assert np.allclose(gt, order_ab, rtol=1e-6, atol=1e-3) or np.allclose(gt, order_ba, rtol=1e-6, atol=1e-3)


def test_data3d_missing_subject_raises(tmp_path):
    pkl = tmp_path / "fake.pkl"
    _write_pkl(pkl, "test", ["s_09_act_02_subact_01_ca_01"] * 2)
    _write_data3d_npz(
        tmp_path / "data_3d_h36m.npz",
        {"S1": {"Directions 1": np.zeros((2, N_JOINTS, 3), dtype=np.float32)}},
    )
    with pytest.raises(KeyError, match="no subject S9"):
        build_true_gt_from_data3d_npz(
            tmp_path / "data_3d_h36m.npz", pkl, "test", subject=9, actions=[2]
        )


def test_data3d_too_short_raises(tmp_path):
    pkl = tmp_path / "fake.pkl"
    _write_pkl(pkl, "test", ["s_09_act_02_subact_01_ca_01"] * 10)
    _write_data3d_npz(
        tmp_path / "data_3d_h36m.npz",
        {"S9": {"Directions 1": np.zeros((4, N_JOINTS, 3), dtype=np.float32)}},
    )
    with pytest.raises(ValueError, match="no ordering"):
        build_true_gt_from_data3d_npz(
            tmp_path / "data_3d_h36m.npz", pkl, "test", subject=9, actions=[2]
        )


def test_vp3d_joint_indices_constant():
    """The verified 17-joint subset must stay the standard H36M-17 skeleton."""
    assert tuple(VP3D_JOINT_INDICES) == (0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27)
    assert VP3D_METERS_TO_MM == 1000.0


# ---------------------------------------------------------------------------
# check_true_gt_reprojection
# ---------------------------------------------------------------------------
def _canonical_npz(path, j3d, uv, K, R, t, conf=None):
    kwargs = {}
    if conf is not None:
        kwargs["confidences"] = conf
    np.savez(
        path,
        points_2d=uv,
        joints_3d=j3d,
        camera_K=K,
        camera_R=R,
        camera_t=t,
        **kwargs,
    )


def _make_scene(n_frames=6, n_joints=17):
    rng = np.random.default_rng(42)
    K = np.array([[1000.0, 0, 512.0], [0, 1000.0, 512.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 5000.0])
    j3d = rng.normal(scale=300.0, size=(n_frames, n_joints, 3))
    j3d[..., 2] += 0.0  # world z=0 plane; camera looks from z=-5000
    uv, depth = reproject(j3d, K, R, t)
    assert np.all(depth > 0)
    # Canonical points_2d layout is (T, V, J, 2) with a single view here.
    return K, R, t, j3d, uv[:, None, :, :]


def test_reprojection_perfect_match(tmp_path):
    K, R, t, j3d, uv = _make_scene()
    path = tmp_path / "scene.npz"
    _canonical_npz(path, j3d, uv, K[None], R[None], t[None])
    overall, frac, per_view = check_npz(path)
    assert overall == pytest.approx(0.0, abs=1e-6)
    assert frac == pytest.approx(1.0)
    assert len(per_view) == 1


def test_reprojection_detects_wrong_labels(tmp_path):
    K, R, t, j3d, uv = _make_scene()
    bad_j3d = j3d + np.array([800.0, 0.0, 0.0])  # gross misalignment
    path = tmp_path / "bad.npz"
    _canonical_npz(path, bad_j3d, uv, K[None], R[None], t[None])
    overall, frac, _ = check_npz(path)
    assert overall > 15.0
    assert frac < 0.5


def test_reprojection_respects_confidence_mask(tmp_path):
    K, R, t, j3d, uv = _make_scene()
    bad_uv = uv.copy()
    bad_uv[:, 0, 3] += 500.0  # corrupt joint 3 of view 0
    conf = np.ones(bad_uv.shape[:3])
    conf[:, 0, 3] = 0.0  # mask the corrupted joint out
    path = tmp_path / "masked.npz"
    _canonical_npz(path, j3d, bad_uv, K[None], R[None], t[None], conf=conf)
    overall, _, _ = check_npz(path)
    assert overall == pytest.approx(0.0, abs=1e-6)


def test_reprojection_missing_keys(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, points_2d=np.zeros((2, 1, 3, 2)))
    with pytest.raises(KeyError, match="missing keys"):
        check_npz(path)
