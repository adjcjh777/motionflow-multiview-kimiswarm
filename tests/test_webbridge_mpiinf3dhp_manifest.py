"""CPU smoke test for the WebBridge MPI-INF-3DHP data audit script.

This test creates a tiny canonical ``.npz`` in a temporary directory, runs the
audit logic, and checks that the generated manifest contains the expected
metadata and coverage gaps.
"""

import numpy as np
import pytest

from experiments.prototypes.audit_webbridge_mpiinf3dhp_data import (
    FileRecord,
    build_coverage,
    build_manifest,
    inspect_npz,
    scan_data_dir,
)


def _make_valid_npz(path, n_frames=10, n_views=14, n_joints=28):
    np.savez(
        path,
        points_2d=np.zeros((n_frames, n_views, n_joints, 2), dtype=np.float64),
        confidences=np.ones((n_frames, n_views, n_joints), dtype=np.float32),
        joints_3d=np.zeros((n_frames, n_joints, 3), dtype=np.float64),
        camera_K=np.tile(np.eye(3, dtype=np.float64)[None, ...], (n_views, 1, 1)),
        camera_R=np.tile(np.eye(3, dtype=np.float64)[None, ...], (n_views, 1, 1)),
        camera_t=np.zeros((n_views, 3), dtype=np.float64),
    )


def test_inspect_valid_npz(tmp_path):
    path = tmp_path / "s_01_seq_01_v14_multiview_m.npz"
    _make_valid_npz(path)
    record = inspect_npz(path)
    assert isinstance(record, FileRecord)
    assert record.is_valid
    assert record.subject == 1
    assert record.seq == "01"
    assert record.n_views_name == 14
    assert record.n_views_actual == 14
    assert record.n_frames == 10
    assert record.n_joints == 28
    assert "m" in record.tags


def test_inspect_invalid_npz(tmp_path):
    path = tmp_path / "s_01_seq_01_v14_multiview_m.npz"
    np.savez(path, points_2d=np.zeros((5, 14, 28, 2)))
    record = inspect_npz(path)
    assert not record.is_valid
    assert any("Missing required key" in e for e in record.errors)


def test_scan_data_dir(tmp_path):
    _make_valid_npz(tmp_path / "s_01_seq_01_v14_multiview_m.npz")
    _make_valid_npz(tmp_path / "s_01_seq_02_v14_multiview_m.npz")
    records = scan_data_dir(tmp_path)
    assert len(records) == 2
    assert all(r.is_valid for r in records)


def test_coverage_and_manifest(tmp_path):
    _make_valid_npz(tmp_path / "s_01_seq_01_v14_multiview_m.npz", n_frames=20)
    _make_valid_npz(tmp_path / "s_01_seq_01_v14_multiview_m_smoke.npz", n_frames=5)
    _make_valid_npz(tmp_path / "s_01_seq_02_v4_multiview_m.npz", n_views=4, n_frames=20)
    # Combined sequence file should cover both S2 seq01 and seq02.
    _make_valid_npz(tmp_path / "s_02_seq_01_02_v14_multiview_m.npz", n_frames=15)
    records = scan_data_dir(tmp_path)
    manifest = build_manifest(tmp_path, records)

    assert manifest["summary"]["total_files"] == 4
    assert manifest["summary"]["valid_files"] == 4
    assert manifest["summary"]["total_frames"] == 20 + 5 + 20 + 15

    by_view = manifest["summary"]["files_by_view"]
    assert by_view["v14"] == 3
    assert by_view["v4"] == 1

    by_tag = manifest["summary"]["files_by_tag"]
    assert by_tag["m"] == 4
    assert by_tag["smoke"] == 1

    coverage = manifest["coverage"]
    cov_s1_seq1 = next(
        c for c in coverage if c["subject"] == 1 and c["seq"] == "01"
    )
    assert cov_s1_seq1["full_v14_m_present"]
    assert cov_s1_seq1["smoke_present"]

    cov_s1_seq2 = next(
        c for c in coverage if c["subject"] == 1 and c["seq"] == "02"
    )
    assert not cov_s1_seq2["full_v14_m_present"]
    assert cov_s1_seq2["v4_present"]

    # S2 seq01 and seq02 should both be covered by the combined file.
    cov_s2_seq1 = next(
        c for c in coverage if c["subject"] == 2 and c["seq"] == "01"
    )
    cov_s2_seq2 = next(
        c for c in coverage if c["subject"] == 2 and c["seq"] == "02"
    )
    assert cov_s2_seq1["full_v14_m_present"]
    assert cov_s2_seq2["full_v14_m_present"]

    missing_subjects = {m["subject"] for m in manifest["missing"] if m["kind"] == "train"}
    assert 2 not in missing_subjects

    test_missing = [m for m in manifest["missing"] if m["kind"] == "test"]
    assert len(test_missing) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
