#!/usr/bin/env python3
"""Read-only quality audit of local WebBridge canonical ``.npz`` files.

Scans ``data/webbridge`` (or a custom root), opens every ``.npz`` file in
memory-mapped mode, and produces a Markdown report with shape/dtype checks,
canonical-format compliance, NaN/Inf/out-of-bounds statistics, and a short
per-file status table.

Usage
-----
    python experiments/audit_webbridge_npz.py
    python experiments/audit_webbridge_npz.py --root data/webbridge --report docs/swarm_iter_next/webbridge_npz_quality_report.md

The script only reads data; it never modifies the underlying ``.npz`` files.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("data/webbridge")
DEFAULT_REPORT = Path("docs/swarm_iter_next/webbridge_npz_quality_report.md")

CANONICAL_KEYS = {
    "points_2d",
    "confidences",
    "joints_3d",
    "camera_K",
    "camera_R",
    "camera_t",
}


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if isinstance(n, float) else f"{n} {unit}"
        n = n / 1024
    return f"{n} B"


def audit_file(path: Path) -> Dict:
    """Return a dict describing the health of one canonical ``.npz`` file."""
    info: Dict = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "keys": [],
        "missing_keys": set(CANONICAL_KEYS),
        "extra_keys": [],
        "shape_ok": False,
        "errors": [],
        "warnings": [],
        "T": None,
        "V": None,
        "J": None,
        "nan_frac": 0.0,
        "inf_frac": 0.0,
        "zero_conf_frac": 0.0,
        "joints_3d_finite": True,
        "status": "OK",
    }

    try:
        with np.load(path, mmap_mode="r") as npz:
            keys = set(npz.files)
            info["keys"] = sorted(keys)
            info["missing_keys"] = sorted(CANONICAL_KEYS - keys)
            info["extra_keys"] = sorted(keys - CANONICAL_KEYS)

            # Helper to fetch shapes/dtypes without pulling everything into RAM.
            shapes: Dict[str, Tuple[int, ...]] = {}
            dtypes: Dict[str, np.dtype] = {}
            for k in keys:
                arr = npz[k]
                shapes[k] = arr.shape
                dtypes[k] = arr.dtype
            info["shapes"] = shapes
            info["dtypes"] = {k: str(v) for k, v in dtypes.items()}

            # Basic shape compliance.
            if not info["missing_keys"]:
                try:
                    T, V, J = shapes["points_2d"][0], shapes["points_2d"][1], shapes["points_2d"][2]
                    info["T"], info["V"], info["J"] = T, V, J
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
                            info["errors"].append(f"{k} shape {shapes.get(k)} != expected {exp}")
                            info["shape_ok"] = False
                    else:
                        if not info["errors"]:
                            info["shape_ok"] = True
                except Exception as exc:  # pragma: no cover - defensive
                    info["errors"].append(f"shape validation failed: {exc}")

            # Numeric quality checks (read-only, on disk-backed arrays).
            if info["shape_ok"]:
                p2d = npz["points_2d"]
                conf = npz["confidences"]
                j3d = npz["joints_3d"]
                total_p2d = p2d.size
                total_conf = conf.size

                nan_p2d = np.isnan(p2d).sum()
                nan_conf = np.isnan(conf).sum()
                nan_j3d = np.isnan(j3d).sum()
                nan_total = nan_p2d + nan_conf + nan_j3d
                inf_p2d = np.isinf(p2d).sum()
                inf_conf = np.isinf(conf).sum()
                inf_j3d = np.isinf(j3d).sum()
                inf_total = inf_p2d + inf_conf + inf_j3d

                info["nan_frac"] = float(nan_total) / max(total_p2d + total_conf + j3d.size, 1)
                info["inf_frac"] = float(inf_total) / max(total_p2d + total_conf + j3d.size, 1)
                info["zero_conf_frac"] = float(np.sum(conf == 0)) / max(total_conf, 1)
                info["joints_3d_finite"] = bool(np.isfinite(j3d).all())

                # Heuristic warnings.
                if info["nan_frac"] > 0:
                    info["warnings"].append(f"NaN fraction {info['nan_frac']:.4e}")
                if info["inf_frac"] > 0:
                    info["warnings"].append(f"Inf fraction {info['inf_frac']:.4e}")
                if info["zero_conf_frac"] > 0.5:
                    info["warnings"].append(f"zero-confidence fraction {info['zero_conf_frac']:.2%}")
                if T < 2:
                    info["warnings"].append("very short sequence (T<2)")
                if V < 2:
                    info["warnings"].append("fewer than 2 views")

        # Derive an overall status.
        if info["errors"]:
            info["status"] = "ERROR"
        elif info["warnings"]:
            info["status"] = "WARN"
        elif info["missing_keys"]:
            info["status"] = "NON-CANONICAL"
        return info

    except Exception as exc:
        info["errors"].append(f"failed to load: {exc}")
        info["status"] = "LOAD-FAIL"
        return info


def _dataset_label(path: Path) -> str:
    """Infer dataset label from the relative path under the WebBridge root."""
    parts = path.parts
    # data/webbridge/<dataset>/.../file.npz -> use second directory component.
    try:
        idx = parts.index("data") if "data" in parts else -1
        if idx >= 0 and len(parts) > idx + 1:
            return parts[idx + 1]
    except Exception:
        pass
    return path.parent.name


def build_report(root: Path, records: List[Dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Aggregate per-dataset counts.
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_dataset[_dataset_label(Path(r["path"]))].append(r)

    status_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        status_counts[r["status"]] += 1

    total_files = len(records)
    total_size = sum(r["size_bytes"] for r in records)
    ok_files = sum(1 for r in records if r["status"] == "OK")

    lines: List[str] = [
        "# WebBridge `.npz` Quality Audit Report\n",
        f"**Date:** {now}\n",
        f"**Root:** `{root.resolve()}`\n",
        f"**Audit script:** `experiments/audit_webbridge_npz.py`\n",
        "**Constraint:** read-only; no files were modified.\n",
    ]

    # Summary
    lines.extend([
        "## 1. Summary\n",
        f"- Total ``.npz`` files scanned: **{total_files}**",
        f"- Total size: **{_human_bytes(total_size)}**",
        f"- Fully canonical / healthy: **{ok_files}**",
        "- Status distribution:",
    ])
    for status, count in sorted(status_counts.items()):
        lines.append(f"  - `{status}`: {count}")
    lines.append("")

    # Per-dataset summary
    lines.append("## 2. Per-Dataset Summary\n")
    lines.append("| Dataset | Files | OK | Warnings/Non-canonical | Total size |")
    lines.append("|---------|-------|----|------------------------|-----------|")
    for ds, recs in sorted(by_dataset.items()):
        n = len(recs)
        ok = sum(1 for r in recs if r["status"] == "OK")
        warn = n - ok
        size = sum(r["size_bytes"] for r in recs)
        lines.append(f"| {ds} | {n} | {ok} | {warn} | {_human_bytes(size)} |")
    lines.append("")

    # Smoke files specifically
    smoke_records = [r for r in records if "smoke" in Path(r["path"]).name]
    lines.append("## 3. Smoke Files\n")
    if smoke_records:
        lines.append("| File | Status | T | V | J | NaN frac | Zero-conf frac | Notes |")
        lines.append("|------|--------|---|---|---|----------|----------------|-------|")
        for r in smoke_records:
            notes = "; ".join(r["errors"] + r["warnings"]) or "—"
            lines.append(
                f"| `{Path(r['path']).name}` | `{r['status']}` | "
                f"{r['T']} | {r['V']} | {r['J']} | "
                f"{r['nan_frac']:.2e} | {r['zero_conf_frac']:.2%} | {notes} |"
            )
    else:
        lines.append("No files matching ``*smoke*.npz`` were found.\n")

    # Full per-file table (truncated in report, but we keep it all for now)
    lines.append("\n## 4. Full Per-File Status\n")
    lines.append("| Path | Status | T | V | J | Missing keys | Notes |")
    lines.append("|------|--------|---|---|---|--------------|-------|")
    for r in records:
        rel = Path(r["path"]).relative_to(root)
        missing = ", ".join(r["missing_keys"]) or "—"
        notes = "; ".join(r["errors"] + r["warnings"]) or "—"
        lines.append(
            f"| `{rel}` | `{r['status']}` | {r['T']} | {r['V']} | {r['J']} | {missing} | {notes} |"
        )

    # Issues & recommendations
    lines.append("\n## 5. Issues and Recommendations\n")
    non_canonical = [r for r in records if r["missing_keys"]]
    errors = [r for r in records if r["status"] in {"ERROR", "LOAD-FAIL"}]
    if errors:
        lines.append(f"- **{len(errors)}** file(s) failed validation or could not be loaded. Review the per-file table above.")
    else:
        lines.append("- No load/validation errors detected.")
    if non_canonical:
        lines.append(f"- **{len(non_canonical)}** file(s) are missing one or more canonical keys; they may be raw/preprocessed artifacts rather than canonical WebBridge ``.npz`` files.")
    else:
        lines.append("- All scanned files contain the canonical keys.")
    if any(r["nan_frac"] > 0 or r["inf_frac"] > 0 for r in records):
        lines.append("- Some files contain NaN or Inf values. Confirm whether these represent legitimate occlusion markers or data corruption.")
    else:
        lines.append("- No NaN/Inf values observed in canonical arrays.")
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only quality audit for local WebBridge canonical .npz files."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="WebBridge data root.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Output Markdown report path.")
    parser.add_argument("--max-files", type=int, default=None, help="Stop after scanning N files (smoke test).")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"Error: root does not exist: {root}", file=sys.stderr)
        return 1

    npz_paths = sorted(root.rglob("*.npz"))
    if not npz_paths:
        print(f"No .npz files found under {root}", file=sys.stderr)
        return 1

    if args.max_files:
        npz_paths = npz_paths[: args.max_files]

    print(f"Scanning {len(npz_paths)} .npz files under {root} ...")
    records: List[Dict] = [audit_file(p) for p in npz_paths]

    report = build_report(root, records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
