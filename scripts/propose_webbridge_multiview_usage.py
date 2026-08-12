#!/usr/bin/env python3
"""Propose a WebBridge mixed multi-view training split.

Scans the canonical ``.npz`` files under ``data/webbridge`` (or a custom root),
validates the canonical multi-view layout, and writes a YAML split manifest
(``configs/deprecated/circular/splits/webbridge_proposed_mixed.yaml`` by default) together with a
short Markdown report explaining which files were selected and why.

The script is read-only with respect to the dataset; it only inspects file
names, shapes, and canonical keys.

Example
-------
    python scripts/propose_webbridge_multiview_usage.py
    python scripts/propose_webbridge_multiview_usage.py --root data/webbridge --out-yaml configs/deprecated/circular/splits/webbridge_proposed_mixed.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


DEFAULT_ROOT = Path("data/webbridge")
DEFAULT_YAML = Path("configs/deprecated/circular/splits/webbridge_proposed_mixed.yaml")
DEFAULT_REPORT = Path("docs/swarm_iter_next/webbridge_multiview_usage_proposal.md")

CANONICAL_KEYS = {
    "points_2d",
    "confidences",
    "joints_3d",
    "camera_K",
    "camera_R",
    "camera_t",
}

# Directories under ``data/webbridge`` that participate in the mixed loader.
# ``None`` means "classify per-file" (used for shelf/campus).
DATASET_DIRS: Dict[str, Optional[str]] = {
    "h36m_meters": "h36m",
    "h36m_corrected": "h36m",
    "mpi_inf_3dhp": "mpi",
    "aistpp_canonical": "aist",
    "shelf_campus": None,
}


def _label_from_path(rel: Path) -> str:
    top = rel.parts[0]
    label = DATASET_DIRS.get(top)
    if label is not None:
        return label
    if top == "shelf_campus":
        name = rel.name.lower()
        if name.startswith("shelf"):
            return "shelf"
        if name.startswith("campus"):
            return "campus"
    return "unknown"


def _is_meter(path: Path) -> bool:
    """Heuristic: filename contains the meter-unit tag used in this repo."""
    name = path.name
    return "_m.npz" in name or "_multiview_m.npz" in name


def _aist_prefix(name: str) -> str:
    """Strip trailing channel tag from an AIST++ filename.

    >>> _aist_prefix("gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz")
    'gBR_sBM_cAll_d04_mBR0'
    """
    m = re.match(r"(.+?)_ch\d+_multiview", name)
    if m:
        return m.group(1)
    return name


def _split_by_subject(rel: Path) -> Optional[str]:
    """H36M / MPI deterministic split based on subject id."""
    name = rel.name
    m = re.search(r"s_(\d+)", name)
    if not m:
        return None
    subject = int(m.group(1))
    dataset = _label_from_path(rel)
    if dataset == "h36m":
        # Subject 9 is reserved for validation; all other subjects train.
        return "val" if subject == 9 else "train"
    if dataset == "mpi":
        # Subject 2 is reserved for validation; all other subjects train.
        return "val" if subject == 2 else "train"
    return None


def _split_aist(files: List[Path], train_ratio: float = 0.85) -> Tuple[List[Path], List[Path]]:
    """Group AIST files by clip prefix and assign whole prefixes to train/val."""
    prefix_to_files: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        prefix_to_files[_aist_prefix(p.name)].append(p)

    train, val = [], []
    for prefix, group in sorted(prefix_to_files.items()):
        digest = hashlib.sha256(prefix.encode()).hexdigest()
        bucket = int(digest, 16) % 1000 / 1000.0
        if bucket < train_ratio:
            train.extend(group)
        else:
            val.extend(group)
    return train, val


def _split_shelf_campus(file: Path) -> Optional[str]:
    name = file.name.lower()
    if "_train_" in name:
        return "train"
    if "_val_" in name:
        return "val"
    return None


def _audit_file(path: Path) -> Dict:
    """Return a dict with canonical shape info for a single ``.npz`` file."""
    info: Dict = {
        "path": path,
        "rel": path,
        "dataset": "unknown",
        "T": None,
        "V": None,
        "J": None,
        "status": "OK",
        "errors": [],
    }
    try:
        with np.load(path, mmap_mode="r") as npz:
            keys = set(npz.files)
            missing = CANONICAL_KEYS - keys
            if missing:
                info["status"] = "NON-CANONICAL"
                info["errors"].append(f"missing keys {sorted(missing)}")
                return info
            T, V, J = npz["points_2d"].shape[:3]
            info["T"] = int(T)
            info["V"] = int(V)
            info["J"] = int(J)
    except Exception as exc:  # pragma: no cover - defensive
        info["status"] = "LOAD-FAIL"
        info["errors"].append(str(exc))
    return info


def _collect_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for rel_dir, _label in DATASET_DIRS.items():
        data_dir = root / rel_dir
        if not data_dir.is_dir():
            continue
        files.extend(sorted(data_dir.rglob("*.npz")))
    return files


def propose_split(
    root: Path,
    exclude_smoke: bool = True,
    exclude_non_meter: bool = True,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], List[Dict]]:
    """Return train/val file lists plus audit records."""
    files = _collect_files(root)
    records: List[Dict] = []
    train: Dict[str, List[Path]] = defaultdict(list)
    val: Dict[str, List[Path]] = defaultdict(list)

    for path in files:
        rel = path.relative_to(root)
        dataset = _label_from_path(rel)
        if exclude_smoke and "smoke" in path.name:
            continue
        # AIST++ canonical files are already in meters despite lacking the `_m` tag.
        if exclude_non_meter and dataset != "aist" and not _is_meter(path):
            continue

        info = _audit_file(path)
        info["rel"] = rel
        info["dataset"] = dataset
        records.append(info)

        if info["status"] != "OK":
            continue

        split: Optional[str] = None
        if dataset in ("h36m", "mpi"):
            split = _split_by_subject(rel)
        elif dataset == "aist":
            # Handled per-prefix below.
            continue
        elif dataset in ("shelf", "campus"):
            split = _split_shelf_campus(path)
        else:
            continue

        if split == "train":
            train[dataset].append(path)
        elif split == "val":
            val[dataset].append(path)

    # AIST++ needs whole-prefix splitting.
    aist_files = [r["path"] for r in records if r["dataset"] == "aist" and r["status"] == "OK"]
    if aist_files:
        aist_train, aist_val = _split_aist(aist_files)
        train["aist"].extend(aist_train)
        val["aist"].extend(aist_val)

    return train, val, records


def _write_yaml(
    out_path: Path,
    train: Dict[str, List[Path]],
    val: Dict[str, List[Path]],
    root: Path,
) -> None:
    """Write the split manifest in the same layout as existing WebBridge split YAMLs."""
    train_paths: List[str] = []
    train_names: List[str] = []
    val_paths: List[str] = []
    val_names: List[str] = []

    # Deterministic ordering for reproducibility.
    # Paths are written relative to the project root, e.g. data/webbridge/...
    for dataset in sorted(train.keys()):
        for path in sorted(train[dataset], key=lambda p: p.as_posix()):
            train_paths.append(path.relative_to(root.parent.parent).as_posix())
            train_names.append(dataset)
    for dataset in sorted(val.keys()):
        for path in sorted(val[dataset], key=lambda p: p.as_posix()):
            val_paths.append(path.relative_to(root.parent.parent).as_posix())
            val_names.append(dataset)

    manifest = {
        "name": "WebBridge proposed mixed multi-view split (17-joint, meter units)",
        "train_paths": train_paths,
        "train_names": train_names,
        "val_paths": val_paths,
        "val_names": val_names,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)


def _write_report(
    out_path: Path,
    root: Path,
    train: Dict[str, List[Path]],
    val: Dict[str, List[Path]],
    records: List[Dict],
) -> None:
    """Write a concise Markdown proposal report."""
    total_train = sum(len(v) for v in train.values())
    total_val = sum(len(v) for v in val.values())
    bad = [r for r in records if r["status"] != "OK"]
    by_status: Dict[str, int] = defaultdict(int)
    for r in records:
        by_status[r["status"]] += 1

    lines = [
        "# WebBridge Multi-View Usage Proposal\n",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n",
        "**Root:** `data/webbridge`\n",
        "**Constraint:** read-only; no dataset files were modified.\n",
        "## 1. Proposal\n",
        "Use the canonical WebBridge `.npz` files listed below for mixed-dataset "
        "multi-view training. All selected files are in **meter units** and follow "
        "the canonical `(T, V, J, 2)` layout. MPI-INF-3DHP uses its native 28-joint "
        "skeleton and is mapped to the common 17-joint layout by "
        "`WebBridgeCanonical17Dataset`.\n",
    ]

    lines.extend([
        "## 2. Train/Val Splits\n",
        "| Dataset | Train files | Val files | Notes |",
        "|---------|------------|-----------|-------|",
    ])
    for dataset in sorted(set(list(train.keys()) + list(val.keys()))):
        n_train = len(train.get(dataset, []))
        n_val = len(val.get(dataset, []))
        note = {
            "h36m": "Subject 9 reserved for validation.",
            "mpi": "Subject 2 reserved for validation (standard MPI benchmark subject).",
            "aist": "Whole clip prefixes split 85/15 by deterministic hash.",
            "shelf": "Explicit train/val files.",
            "campus": "Explicit train/val files.",
        }.get(dataset, "")
        lines.append(f"| {dataset} | {n_train} | {n_val} | {note} |")

    lines.extend([
        "\n## 3. Data Quality\n",
        f"- Total canonical `.npz` records inspected: **{len(records)}**",
        f"- Status distribution: `{dict(by_status)}`",
    ])
    if bad:
        lines.append(f"- **{len(bad)}** file(s) were excluded due to load/validation issues.")
    else:
        lines.append("- No load/validation issues detected among inspected files.")

    lines.extend([
        "\n## 4. Next Steps\n",
        "1. Run the loader smoke test with the generated YAML:\n",
        "   ```bash\n",
        "   python -m motionflow_mv.data.webbridge_mixed_dataset --smoke\n",
        "   ```\n",
        "2. If `OmniMultiViewFusionV2`/`V3` rejects variable view counts, add a "
        "   `view_mask` so that padded 14-view slots are ignored.\n",
        "3. Train a small mixed-dataset model and measure cross-dataset MPJPE on "
        "   H36M subject 9 and MPI subject 2 before scaling up to A800.\n",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propose a WebBridge mixed multi-view training split."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="WebBridge data root.")
    parser.add_argument("--out-yaml", type=Path, default=DEFAULT_YAML, help="Output YAML split path.")
    parser.add_argument("--out-report", type=Path, default=DEFAULT_REPORT, help="Output Markdown report path.")
    parser.add_argument("--include-smoke", action="store_true", help="Include *_smoke*.npz files.")
    parser.add_argument("--include-non-meter", action="store_true", help="Include non-meter-unit files (risky for mixed training).")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"Error: root does not exist: {root}", file=sys.stderr)
        return 1

    train, val, records = propose_split(
        root,
        exclude_smoke=not args.include_smoke,
        exclude_non_meter=not args.include_non_meter,
    )

    _write_yaml(args.out_yaml, train, val, root)
    _write_report(args.out_report, root, train, val, records)

    print(f"YAML split written to: {args.out_yaml}")
    print(f"Report written to: {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
