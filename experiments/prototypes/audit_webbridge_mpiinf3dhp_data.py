#!/usr/bin/env python3
"""Audit WebBridge MPI-INF-3DHP canonical data and emit a machine-readable manifest.

The script scans ``data/webbridge/mpi_inf_3dhp`` (or any user-supplied directory)
for canonical ``.npz`` files, validates their layout against the WebBridge
specification, and reports which subjects/sequences of the official
MPI-INF-3DHP release are available locally.

Typical usage::

    python experiments/prototypes/audit_webbridge_mpiinf3dhp_data.py \
        --data_dir data/webbridge/mpi_inf_3dhp \
        --out outputs/webbridge_mpi_inf_3dhp_manifest.json \
        --csv outputs/webbridge_mpi_inf_3dhp_manifest.csv

The manifest is intentionally read-only: it does not download or convert data.
"""

import argparse
import csv
import datetime
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Official MPI-INF-3DHP release structure (from README.txt):
# 8 training subjects, 2 sequences each; 6 test sequences.
TRAIN_SUBJECTS = list(range(1, 9))
TRAIN_SEQUENCES = [1, 2]
TEST_SEQUENCES = ["TS1", "TS2", "TS3", "TS4", "TS5", "TS6"]

# Keys required by the WebBridge mixed-dataset loader.
REQUIRED_KEYS = ("points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t")

# Filename components we recognise.
NAME_RE = re.compile(
    r"^s_(?P<subject>\d{2})_"
    r"seq_(?P<seq>\d{2}(?:_\d{2})?)_"
    r"(?P<views>v\d+)_"
    r"multiview"
    r"(?P<extras>(?:_[a-z]+)*)"
    r"\.npz$"
)


@dataclass(frozen=True)
class FileRecord:
    """Metadata for one canonical ``.npz`` file."""

    path: str
    filename: str
    subject: int
    seq: str
    n_views_name: int
    tags: Tuple[str, ...]
    file_size_bytes: int
    n_frames: int
    n_views_actual: int
    n_joints: int
    key_shapes: Dict[str, List[int]]
    key_dtypes: Dict[str, str]
    errors: List[str]
    is_valid: bool


def parse_filename(filename: str) -> Optional[Dict]:
    """Extract subject/seq/view/tags from a canonical npz filename."""
    m = NAME_RE.match(filename)
    if not m:
        return None
    extras = tuple(x for x in m.group("extras").strip("_").split("_") if x)
    return {
        "subject": int(m.group("subject")),
        "seq": m.group("seq"),
        "n_views_name": int(m.group("views").lstrip("v")),
        "tags": extras,
    }


def inspect_npz(path: Path) -> FileRecord:
    """Open a single ``.npz`` and return its metadata/validation status."""
    filename = path.name
    parsed = parse_filename(filename)
    if parsed is None:
        parsed = {
            "subject": -1,
            "seq": "unknown",
            "n_views_name": -1,
            "tags": (),
        }
        error = f"Filename does not match expected MPI-INF-3DHP pattern: {filename}"
    else:
        error = None

    errors: List[str] = []
    if error:
        errors.append(error)

    key_shapes: Dict[str, List[int]] = {}
    key_dtypes: Dict[str, str] = {}
    n_frames = -1
    n_views_actual = -1
    n_joints = -1

    try:
        with np.load(path) as data:
            for key in REQUIRED_KEYS:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
                else:
                    arr = data[key]
                    key_shapes[key] = list(arr.shape)
                    key_dtypes[key] = str(arr.dtype)

            if "points_2d" in data:
                n_frames = int(data["points_2d"].shape[0])
                n_views_actual = int(data["points_2d"].shape[1])
                n_joints = int(data["points_2d"].shape[2])

            if (
                "camera_K" in data
                and n_views_actual != -1
                and int(data["camera_K"].shape[0]) != n_views_actual
            ):
                errors.append(
                    f"camera_K view count {data['camera_K'].shape[0]} does not match "
                    f"points_2d view count {n_views_actual}"
                )
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"Failed to read npz: {exc}")

    is_valid = len(errors) == 0 and n_frames > 0 and n_views_actual > 0
    return FileRecord(
        path=str(path),
        filename=filename,
        subject=parsed["subject"],
        seq=parsed["seq"],
        n_views_name=parsed["n_views_name"],
        tags=parsed["tags"],
        file_size_bytes=path.stat().st_size,
        n_frames=n_frames,
        n_views_actual=n_views_actual,
        n_joints=n_joints,
        key_shapes=key_shapes,
        key_dtypes=key_dtypes,
        errors=errors,
        is_valid=is_valid,
    )


def scan_data_dir(data_dir: Path) -> List[FileRecord]:
    """Scan *data_dir* for ``.npz`` files and inspect each one."""
    records: List[FileRecord] = []
    for path in sorted(data_dir.glob("*.npz"), key=lambda p: p.name):
        if not path.is_file():
            continue
        records.append(inspect_npz(path))
    return records


def _seq_matches(record: FileRecord, seq: str) -> bool:
    """Check whether *record* covers an individual sequence *seq*.

    A record with ``seq == '01_02'`` is considered to cover both ``'01'`` and
    ``'02'``.  Otherwise an exact match is required.
    """
    if record.seq == seq:
        return True
    parts = record.seq.split("_")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return seq in parts
    return False


def _has_full_record(records: List[FileRecord], subject: int, seq: str) -> bool:
    """Return ``True`` if a non-smoke, meters, v14 file exists for the train seq."""
    for r in records:
        if r.subject != subject:
            continue
        if not _seq_matches(r, seq):
            continue
        if r.n_views_name == 14 and "m" in r.tags and "smoke" not in r.tags:
            return True
    return False


def _has_smoke_record(records: List[FileRecord], subject: int, seq: str) -> bool:
    for r in records:
        if r.subject == subject and _seq_matches(r, seq) and "smoke" in r.tags:
            return True
    return False


def _has_v4_record(records: List[FileRecord], subject: int, seq: str) -> bool:
    for r in records:
        if r.subject == subject and _seq_matches(r, seq) and r.n_views_name == 4:
            return True
    return False


def build_coverage(records: List[FileRecord]) -> Tuple[List[Dict], List[Dict]]:
    """Return per-train-sequence coverage and a list of missing items."""
    coverage = []
    missing = []
    for subject in TRAIN_SUBJECTS:
        for seq_int in TRAIN_SEQUENCES:
            seq = f"{seq_int:02d}"
            full = _has_full_record(records, subject, seq)
            smoke = _has_smoke_record(records, subject, seq)
            v4 = _has_v4_record(records, subject, seq)
            present_files = [
                r.filename
                for r in records
                if r.subject == subject and r.seq == seq and r.is_valid
            ]
            entry = {
                "subject": subject,
                "seq": seq,
                "full_v14_m_present": full,
                "smoke_present": smoke,
                "v4_present": v4,
                "present_files": present_files,
            }
            coverage.append(entry)
            if not full:
                missing.append(
                    {
                        "kind": "train",
                        "subject": subject,
                        "seq": seq,
                        "expected_file": f"s_{subject:02d}_seq_{seq}_v14_multiview_m.npz",
                    }
                )

    for test_seq in TEST_SEQUENCES:
        missing.append(
            {
                "kind": "test",
                "subject": None,
                "seq": test_seq,
                "expected_file": f"{test_seq}_v14_multiview_m.npz",
            }
        )

    return coverage, missing


def build_manifest(
    data_dir: Path,
    records: List[FileRecord],
) -> Dict:
    """Assemble the JSON manifest dictionary."""
    coverage, missing = build_coverage(records)

    total_size = sum(r.file_size_bytes for r in records)
    total_frames = sum(r.n_frames for r in records)

    view_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}
    for r in records:
        view_key = f"v{r.n_views_name}" if r.n_views_name > 0 else "unknown"
        view_counts[view_key] = view_counts.get(view_key, 0) + 1
        for tag in r.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    raw_zip = data_dir / "mpi_inf_3dhp.zip"
    raw_dir = data_dir / "mpi_inf_3dhp"

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "summary": {
            "total_files": len(records),
            "valid_files": sum(1 for r in records if r.is_valid),
            "invalid_files": sum(1 for r in records if not r.is_valid),
            "total_frames": total_frames,
            "total_size_bytes": total_size,
            "files_by_view": view_counts,
            "files_by_tag": tag_counts,
            "raw_archive_present": raw_zip.is_file(),
            "raw_extracted_dir_present": raw_dir.is_dir(),
        },
        "files": [asdict(r) for r in records],
        "coverage": coverage,
        "missing": missing,
    }


def write_csv(manifest: Dict, csv_path: Path) -> None:
    """Write a flat CSV representation of the file list."""
    rows = manifest["files"]
    if not rows:
        csv_path.write_text("")
        return

    fieldnames = [
        "filename",
        "subject",
        "seq",
        "n_views_name",
        "tags",
        "n_frames",
        "n_views_actual",
        "n_joints",
        "file_size_bytes",
        "is_valid",
        "errors",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "subject": row["subject"],
                    "seq": row["seq"],
                    "n_views_name": row["n_views_name"],
                    "tags": " ".join(row["tags"]),
                    "n_frames": row["n_frames"],
                    "n_views_actual": row["n_views_actual"],
                    "n_joints": row["n_joints"],
                    "file_size_bytes": row["file_size_bytes"],
                    "is_valid": row["is_valid"],
                    "errors": "; ".join(row["errors"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit WebBridge MPI-INF-3DHP canonical npz availability."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp"),
        help="Directory containing canonical ``.npz`` files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/webbridge_mpi_inf_3dhp_manifest.json"),
        help="Destination JSON manifest path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional flat CSV manifest path.",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    records = scan_data_dir(args.data_dir)
    manifest = build_manifest(args.data_dir, records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote JSON manifest to {args.out}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(manifest, args.csv)
        print(f"Wrote CSV manifest to {args.csv}")

    print(
        f"Scanned {manifest['summary']['total_files']} files; "
        f"{manifest['summary']['valid_files']} valid, "
        f"{len(manifest['missing'])} expected sequences missing."
    )


if __name__ == "__main__":
    main()
