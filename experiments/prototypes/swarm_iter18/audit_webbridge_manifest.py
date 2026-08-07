#!/usr/bin/env python3
"""P07 WebBridge data manifest auditor.

Scans the local WebBridge canonical ``.npz`` files for the datasets required by
the ICRA/CVPR 2027 multi-view experiments (MPI-INF-3DHP, H36M, AIST++,
Shelf/Campus) and writes a Markdown manifest plus a machine-readable JSON.

Usage::

    python experiments/prototypes/swarm_iter18/audit_webbridge_manifest.py
    python experiments/prototypes/swarm_iter18/audit_webbridge_manifest.py \
        --root data/webbridge \
        --md docs/swarm_iter18/P07_webbridge_manifest.md \
        --json outputs/swarm_iter18/P07_webbridge_manifest.json

The script is read-only: it never modifies the underlying data files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("data/webbridge")
DEFAULT_MD = Path("docs/swarm_iter18/P07_webbridge_manifest.md")
DEFAULT_JSON = Path("outputs/swarm_iter18/P07_webbridge_manifest.json")

CANONICAL_KEYS = {
    "points_2d",
    "confidences",
    "joints_3d",
    "camera_K",
    "camera_R",
    "camera_t",
}

DATASETS: List[Tuple[str, str, str]] = [
    ("MPI-INF-3DHP", "mpi_inf_3dhp", "MPI-INF-3DHP canonical multi-view npz files"),
    ("H36M", "h36m_meters", "Human3.6M canonical multi-view npz files (meters)"),
    ("AIST++", "aistpp_canonical", "AIST++ canonical multi-view npz files"),
    ("Shelf/Campus", "shelf_campus", "Shelf and Campus canonical multi-view npz files"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def audit_file(path: Path) -> Dict:
    """Return a dict describing the health of one canonical ``.npz`` file."""
    info: Dict = {
        "path": str(path),
        "rel_path": str(path.relative_to(path.parent.parent)),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "keys": [],
        "missing_keys": [],
        "T": None,
        "V": None,
        "J": None,
        "shape_ok": False,
        "errors": [],
        "warnings": [],
        "status": "OK",
    }

    try:
        with np.load(path, mmap_mode="r") as npz:
            keys = set(npz.files)
            info["keys"] = sorted(keys)
            info["missing_keys"] = sorted(CANONICAL_KEYS - keys)

            shapes: Dict[str, Tuple[int, ...]] = {}
            for k in keys:
                shapes[k] = npz[k].shape

            if not info["missing_keys"]:
                try:
                    T, V, J = (
                        shapes["points_2d"][0],
                        shapes["points_2d"][1],
                        shapes["points_2d"][2],
                    )
                    info["T"] = int(T)
                    info["V"] = int(V)
                    info["J"] = int(J)
                    expected = {
                        "points_2d": (T, V, J, 2),
                        "confidences": (T, V, J),
                        "joints_3d": (T, J, 3),
                        "camera_K": (V, 3, 3),
                        "camera_R": (V, 3, 3),
                        "camera_t": (V, 3),
                    }
                    for k, exp in expected.items():
                        if shapes.get(k) != exp:
                            info["errors"].append(
                                f"{k} shape {shapes.get(k)} != expected {exp}"
                            )
                    info["shape_ok"] = not info["errors"]
                except Exception as exc:
                    info["errors"].append(f"shape validation failed: {exc}")
                    info["shape_ok"] = False
    except Exception as exc:
        info["errors"].append(f"failed to load: {exc}")

    if info["errors"]:
        info["status"] = "ERROR"
    elif info["missing_keys"]:
        info["status"] = "NON-CANONICAL"
    elif not info["shape_ok"]:
        info["status"] = "ERROR"

    return info


# ---------------------------------------------------------------------------
# Per-dataset aggregation helpers
# ---------------------------------------------------------------------------


def _mpi_coverage(records: List[Dict]) -> List[str]:
    """Return a short list of present MPI-INF-3DHP subjects/sequences."""
    present = set()
    for rec in records:
        m = re.search(r"s_(\d+)_seq_(\d+)", rec["filename"])
        if m:
            present.add((int(m.group(1)), m.group(2)))
    lines = []
    for subj in sorted({s for s, _ in present}):
        seqs = sorted({seq for s, seq in present if s == subj})
        lines.append(f"- Subject {subj}: sequences {', '.join(seqs)}")
    return lines


def _h36m_coverage(records: List[Dict]) -> List[str]:
    """Return a short list of present H36M subjects/actions."""
    present: Dict[int, List[int]] = defaultdict(list)
    for rec in records:
        m = re.search(r"s_(\d+)_acts_(\d+)", rec["filename"])
        if m:
            present[int(m.group(1))].append(int(m.group(2)))
    lines = []
    for subj in sorted(present):
        acts = sorted(set(present[subj]))
        lines.append(f"- Subject {subj}: {len(acts)} action files ({acts[0]}–{acts[-1]})")
    return lines


def _aist_groups(records: List[Dict]) -> Dict[str, Dict]:
    """Group AIST++ records by clip prefix and count channels/frames."""
    groups: Dict[str, Dict] = {}
    for rec in records:
        m = re.match(
            r"(g[A-Z]+_s[A-Z]+_cAll_d\d+_m[A-Z0-9]+)_ch\d+_multiview\.npz",
            rec["filename"],
        )
        prefix = m.group(1) if m else rec["filename"].rsplit("_ch", 1)[0]
        if prefix not in groups:
            groups[prefix] = {
                "files": [],
                "T": rec["T"],
                "V": rec["V"],
                "J": rec["J"],
            }
        groups[prefix]["files"].append(rec)
    return groups


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _header() -> List[str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return [
        "# P07 WebBridge Data Manifest\n",
        f"**Generated:** {now}\n",
        "**Branch:** `feat/swarm-iter18-omniview`\n",
        "**Author:** Kimi Code subagent\n",
        "**Root:** `data/webbridge`\n",
        "**Constraint:** read-only; no files were modified.\n",
    ]


def _dataset_summary_table(dataset_results: Dict[str, List[Dict]]) -> List[str]:
    lines = [
        "## 1. Summary\n",
        "| Dataset | Files | OK | Errors | Total size | Total frames | Views | Joints | Notes |",
        "|---------|-------|----|--------|-----------|--------------|-------|--------|-------|",
    ]
    grand_total = 0
    for name, _dir, desc in DATASETS:
        records = dataset_results[name]
        if not records:
            lines.append(
                f"| {name} | 0 | 0 | 0 | 0 B | — | — | — | {desc} |"
            )
            continue
        ok = sum(1 for r in records if r["status"] == "OK")
        err = len(records) - ok
        size = sum(r["size_bytes"] for r in records)
        frames = sum(r["T"] for r in records if r["T"] is not None)
        vs = sorted({r["V"] for r in records if r["V"] is not None})
        js = sorted({r["J"] for r in records if r["J"] is not None})
        lines.append(
            f"| {name} | {len(records)} | {ok} | {err} | "
            f"{_human_bytes(size)} | {frames:,} | {', '.join(map(str, vs))} | "
            f"{', '.join(map(str, js))} | {desc} |"
        )
        grand_total += size
    lines.append(f"\n**Total audited storage:** {_human_bytes(grand_total)}\n")
    return lines


def _mpi_section(records: List[Dict]) -> List[str]:
    lines = [
        "## 2. MPI-INF-3DHP\n",
        "Canonical multi-view ``.npz`` files under `data/webbridge/mpi_inf_3dhp`.\n",
        "### 2.1 Coverage\n",
    ]
    lines.extend(_mpi_coverage(records) or ["- No canonical files detected."])
    lines.append("\n### 2.2 Per-file inventory\n")
    lines.append(
        "| File | Status | T | V | J | Size | Notes |"
    )
    lines.append(
        "|------|--------|---|---|---|------|-------|"
    )
    for rec in sorted(records, key=lambda r: r["filename"]):
        notes = "; ".join(rec["errors"] + rec["warnings"]) or "—"
        lines.append(
            f"| `{rec['filename']}` | `{rec['status']}` | "
            f"{rec['T']} | {rec['V']} | {rec['J']} | "
            f"{_human_bytes(rec['size_bytes'])} | {notes} |"
        )
    return lines


def _h36m_section(records: List[Dict]) -> List[str]:
    lines = [
        "\n## 3. Human3.6M\n",
        "Canonical multi-view ``.npz`` files under `data/webbridge/h36m_meters`.\n",
        "### 3.1 Coverage\n",
    ]
    lines.extend(_h36m_coverage(records) or ["- No canonical files detected."])
    lines.append("\n### 3.2 Per-file inventory\n")
    lines.append(
        "| File | Status | T | V | J | Size | Notes |"
    )
    lines.append(
        "|------|--------|---|---|---|------|-------|"
    )
    for rec in sorted(records, key=lambda r: r["filename"]):
        notes = "; ".join(rec["errors"] + rec["warnings"]) or "—"
        lines.append(
            f"| `{rec['filename']}` | `{rec['status']}` | "
            f"{rec['T']} | {rec['V']} | {rec['J']} | "
            f"{_human_bytes(rec['size_bytes'])} | {notes} |"
        )
    return lines


def _aist_section(records: List[Dict]) -> List[str]:
    lines = [
        "\n## 4. AIST++\n",
        "Canonical multi-view ``.npz`` files under `data/webbridge/aistpp_canonical`.\n",
    ]
    if not records:
        lines.append("- No canonical files detected.\n")
        return lines

    groups = _aist_groups(records)
    total_files = len(records)
    total_frames = sum(r["T"] for r in records if r["T"] is not None)
    ok = sum(1 for r in records if r["status"] == "OK")
    err = total_files - ok

    lines.extend([
        f"- **Total files:** {total_files:,}",
        f"- **Unique clips:** {len(groups):,}",
        f"- **Healthy (OK):** {ok:,}",
        f"- **With errors:** {err:,}",
        f"- **Total frames:** {total_frames:,}",
        "- **Shape convention:** each file contains a single camera/channel; "
        "multiple `_chNN` files may be pooled to form a multi-view clip.",
    ])

    # Genre distribution
    genre_counts: Dict[str, int] = defaultdict(int)
    genre_examples: Dict[str, str] = {}
    for rec in records:
        m = re.match(r"g([A-Z]+)", rec["filename"])
        if m:
            genre = m.group(1)
            genre_counts[genre] += 1
            if genre not in genre_examples:
                genre_examples[genre] = rec["filename"]
    lines.append("\n### 4.1 Genre distribution\n")
    lines.append("| Genre | Files | Example |")
    lines.append("|-------|-------|---------|")
    for g, c in sorted(genre_counts.items()):
        lines.append(f"| {g} | {c:,} | `{genre_examples[g]}` |")

    # Per-clip aggregated table (first 20)
    lines.append("\n### 4.2 Per-clip aggregated inventory (first 20 of {:,})\n".format(len(groups)))
    lines.append(
        "| Clip prefix | Files | Total frames | T | V | J | Notes |"
    )
    lines.append(
        "|-------------|-------|--------------|---|---|---|-------|"
    )
    for prefix in sorted(groups)[:20]:
        grp = groups[prefix]
        files = grp["files"]
        total = sum(r["T"] for r in files if r["T"] is not None)
        bad = [r["filename"] for r in files if r["status"] != "OK"]
        notes = f"{len(bad)} errors" if bad else "—"
        lines.append(
            f"| `{prefix}` | {len(files)} | {total:,} | "
            f"{files[0]['T']} | {files[0]['V']} | {files[0]['J']} | {notes} |"
        )
    if len(groups) > 20:
        lines.append(f"\n> {len(groups) - 20:,} additional clips omitted for brevity.\n")

    return lines


def _shelf_campus_section(records: List[Dict]) -> List[str]:
    lines = [
        "\n## 5. Shelf / Campus\n",
        "Canonical multi-view ``.npz`` files under `data/webbridge/shelf_campus`.\n",
    ]
    if not records:
        lines.append("- No canonical files detected.\n")
        return lines
    lines.append(
        "| File | Dataset | Split | Views | T | J | Size | Status | Notes |"
    )
    lines.append(
        "|------|---------|-------|-------|---|---|------|--------|-------|"
    )
    for rec in sorted(records, key=lambda r: r["filename"]):
        name = rec["filename"]
        dataset = "Shelf" if name.startswith("shelf") else "Campus"
        split = "train" if "_train_" in name else "val"
        notes = "; ".join(rec["errors"] + rec["warnings"]) or "—"
        lines.append(
            f"| `{name}` | {dataset} | {split} | {rec['V']} | {rec['T']} | "
            f"{rec['J']} | {_human_bytes(rec['size_bytes'])} | {rec['status']} | {notes} |"
        )
    return lines


def build_markdown(dataset_results: Dict[str, List[Dict]]) -> str:
    lines: List[str] = []
    lines.extend(_header())
    lines.extend(_dataset_summary_table(dataset_results))
    lines.extend(_mpi_section(dataset_results.get("MPI-INF-3DHP", [])))
    lines.extend(_h36m_section(dataset_results.get("H36M", [])))
    lines.extend(_aist_section(dataset_results.get("AIST++", [])))
    lines.extend(_shelf_campus_section(dataset_results.get("Shelf/Campus", [])))

    lines.extend([
        "\n## 6. Notes and blockers\n",
        "- All reported sizes are on-disk ``.npz`` sizes.\n",
        "- `OK` means the file contains all six canonical keys and the expected shapes.\n",
        "- AIST++ files are stored per camera/channel; multi-view clips are built by "
        "grouping ``_chNN`` files with the same clip prefix.\n",
    ])

    error_count = sum(
        1 for recs in dataset_results.values() for r in recs if r["status"] != "OK"
    )
    if error_count:
        lines.append(f"- **{error_count} file(s)** failed validation; see per-file notes above.\n")
    else:
        lines.append("- No validation errors were detected across the audited files.\n")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate P07 WebBridge data manifest."
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="WebBridge data root."
    )
    parser.add_argument(
        "--md", type=Path, default=DEFAULT_MD, help="Output Markdown manifest."
    )
    parser.add_argument(
        "--json", type=Path, default=DEFAULT_JSON, help="Output JSON manifest."
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"Error: root does not exist: {root}", file=sys.stderr)
        return 1

    dataset_results: Dict[str, List[Dict]] = {}
    for name, rel_dir, _ in DATASETS:
        data_dir = root / rel_dir
        if not data_dir.is_dir():
            dataset_results[name] = []
            continue
        npz_paths = sorted(data_dir.rglob("*.npz"))
        print(f"Scanning {name}: {len(npz_paths)} .npz files ...")
        dataset_results[name] = [audit_file(p) for p in npz_paths]

    # Write Markdown
    md = build_markdown(dataset_results)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(md, encoding="utf-8")
    print(f"Markdown manifest written to: {args.md}")

    # Write JSON
    json_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "datasets": {
            name: [r for r in recs] for name, recs in dataset_results.items()
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"JSON manifest written to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
