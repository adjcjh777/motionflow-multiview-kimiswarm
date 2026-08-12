#!/usr/bin/env python3
"""Audit a WebBridge mixed-dataset split manifest.

Scans every ``.npz`` referenced by a YAML split config and reports
inconsistencies that would break the v25 ``MultiViewGeometryFusionV25``
loader or the mixed-dataset training loop.

Examples
--------
    python scripts/audit_mixed_loader.py configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml
    python scripts/audit_mixed_loader.py configs/deprecated/circular/splits/webbridge_all_train.yaml --strict-shape
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


EXPECTED_ARRAYS = (
    "points_2d",
    "confidences",
    "joints_3d",
    "camera_K",
    "camera_R",
    "camera_t",
)


def _infer_dataset_name(path: str, explicit_name: str | None = None) -> str:
    """Infer a dataset tag from an explicit name or the path."""
    if explicit_name:
        return explicit_name
    p = path.lower()
    if "h36m" in p:
        return "h36m"
    if "mpi" in p:
        return "mpi"
    if "aist" in p:
        return "aist"
    if "3dpw" in p:
        return "3dpw"
    if "shelf" in p:
        return "shelf"
    if "campus" in p:
        return "campus"
    return "unknown"


def _looks_like_meters(joints_3d: np.ndarray) -> bool:
    """Heuristic: MPI/H36M in meters have a smaller bounding box than mm."""
    finite = joints_3d[np.isfinite(joints_3d)]
    if finite.size == 0:
        return True  # cannot infer; let other checks catch all-NaN arrays.
    size = np.ptp(finite)  # max - min
    # A person is ~2 m tall; in mm the same range is ~2000.
    return size < 100.0


def audit_file(npz_path: str, name: str | None = None) -> tuple[list[str], dict]:
    """Return issues and file stats for one ``.npz`` file."""
    issues: list[str] = []
    path = Path(npz_path)
    tag = _infer_dataset_name(npz_path, name)

    if not path.exists():
        issues.append(f"{npz_path}: missing file")
        return issues, {"tag": tag, "T": 0, "V": 0, "J": 0}

    try:
        data = np.load(npz_path)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"{npz_path}: cannot load ({exc})")
        return issues, {"tag": tag, "T": 0, "V": 0, "J": 0}

    for arr_name in EXPECTED_ARRAYS:
        if arr_name not in data:
            issues.append(f"{npz_path}: missing array '{arr_name}'")

    try:
        points_2d = data["points_2d"]
        confidences = data["confidences"]
        joints_3d = data["joints_3d"]
        camera_K = data["camera_K"]
        camera_R = data["camera_R"]
        camera_t = data["camera_t"]
    except KeyError:
        return issues, {"tag": tag, "T": 0, "V": 0, "J": 0}

    T = V = J = 0
    if points_2d.ndim != 4:
        issues.append(f"{npz_path}: points_2d rank {points_2d.ndim} != 4")
    else:
        T, V, J, c = points_2d.shape
        if c != 2:
            issues.append(f"{npz_path}: points_2d last dim {c} != 2")
        if confidences.shape != (T, V, J):
            issues.append(
                f"{npz_path}: confidences shape {confidences.shape} "
                f"does not match points_2d (T={T}, V={V}, J={J})"
            )
        if joints_3d.shape[:2] != (T, J):
            issues.append(
                f"{npz_path}: joints_3d shape {joints_3d.shape} "
                f"does not match points_2d (T={T}, J={J})"
            )
        if camera_K.ndim != 3 or camera_K.shape[0] != V:
            issues.append(
                f"{npz_path}: camera_K shape {camera_K.shape} inconsistent with V={V}"
            )
        if camera_R.ndim != 3 or camera_R.shape[0] != V:
            issues.append(
                f"{npz_path}: camera_R shape {camera_R.shape} inconsistent with V={V}"
            )
        if camera_t.ndim != 2 or camera_t.shape[0] != V:
            issues.append(
                f"{npz_path}: camera_t shape {camera_t.shape} inconsistent with V={V}"
            )

    if confidences.ndim != 3:
        issues.append(f"{npz_path}: confidences rank {confidences.ndim} != 3")
    if joints_3d.ndim != 3:
        issues.append(f"{npz_path}: joints_3d rank {joints_3d.ndim} != 3")

    finite_joints = joints_3d[np.isfinite(joints_3d)]
    if not _looks_like_meters(joints_3d):
        ptp = np.ptp(finite_joints) if finite_joints.size else float("nan")
        issues.append(
            f"{npz_path}: joints_3d looks like millimeters "
            f"(range={ptp:.1f}); v25 expects meters"
        )

    # Per-dataset shape expectations for the WebBridge 17-joint mixed loader.
    # These are checks, not hard errors, because some configs intentionally mix
    # native skeletons.
    if tag in {"h36m", "3dpw"} and J != 17:
        issues.append(f"{npz_path}: {tag} expects 17 joints, found {J}")
    if tag == "mpi" and J not in (17, 28):
        issues.append(f"{npz_path}: mpi expects 17 or 28 joints, found {J}")
    if tag == "aist" and J != 17:
        issues.append(f"{npz_path}: {tag} expects 17 joints, found {J}")

    return issues, {"tag": tag, "T": T, "V": V, "J": J}


def _load_manifest(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _manifest_entries(manifest: dict) -> list[tuple[str, str | None]]:
    """Return list of (npz_path, optional_dataset_name)."""
    entries: list[tuple[str, str | None]] = []
    if "train_paths" in manifest and "train_names" in manifest:
        for p, n in zip(manifest["train_paths"], manifest["train_names"]):
            entries.append((p, n))
        for p, n in zip(manifest["val_paths"], manifest["val_names"]):
            entries.append((p, n))
    else:
        for key in ("train", "val", "test"):
            for p in manifest.get(key, []):
                entries.append((p, None))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a WebBridge mixed-dataset split manifest.")
    parser.add_argument("manifest", type=str, help="Path to YAML split manifest.")
    parser.add_argument(
        "--strict-shape",
        action="store_true",
        help="Treat shape/dimension warnings as fatal errors.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    entries = _manifest_entries(manifest)

    if not entries:
        print(f"{args.manifest}: no entries found", file=sys.stderr)
        return 1

    print(f"Auditing {len(entries)} entries from {args.manifest}")

    all_issues: list[str] = []
    stats: list[dict] = []
    for path, name in entries:
        issues, stat = audit_file(path, name)
        all_issues.extend(issues)
        stats.append(stat)

    # Cross-file consistency checks.
    n_views_set = {s["V"] for s in stats}
    n_joints_set = {s["J"] for s in stats}
    if len(n_views_set) > 1:
        all_issues.append(
            f"Mixed n_views across manifest: {sorted(n_views_set)}. "
            "v25 geometry attention expects a single fixed n_views per run."
        )
    if len(n_joints_set) > 1:
        all_issues.append(
            f"Mixed n_joints across manifest: {sorted(n_joints_set)}. "
            "Use WebBridgeCanonical17Dataset for a common 17-joint skeleton."
        )

    for issue in all_issues:
        print("  - " + issue)

    if all_issues:
        print(f"\nFound {len(all_issues)} issue(s).")
        if args.strict_shape:
            return 1
        # By default only missing files / un-loadable files are fatal.
        fatal = [i for i in all_issues if "missing file" in i or "cannot load" in i]
        if fatal:
            return 1
    else:
        print("\nNo issues found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
