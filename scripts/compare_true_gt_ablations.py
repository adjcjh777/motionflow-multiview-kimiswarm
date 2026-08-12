#!/usr/bin/env python3
"""Compare true-GT H36M ablations from A800 (or local) training outputs.

Given a list of output log/checkpoint/directory paths, extracts for each run:

* best validation MPJPE (mm) and the epoch it occurred
* final / latest validation MPJPE (mm)
* test MPJPE (mm) if an eval JSON or test line is present
* training time if it can be inferred from the log or file metadata

The script is designed for the v25 true-GT ablation queue, but is generic
enough for any run that emits ``Epoch N: ... val_MPJPE=XX.XXmm`` lines.

Usage
-----
Local or mounted A800 paths::

    python scripts/compare_true_gt_ablations.py \
        /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_baseline_fix.log \
        /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_geometry_regularization_a800.pth

Fetch from A800 via SSH::

    python scripts/compare_true_gt_ablations.py \
        --ssh-host a800-D \
        --a800-repo /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20 \
        v25_true_gt_baseline_fix v25_true_gt_geometry_regularization_a800

Output is a Markdown table printed to stdout; use ``--csv`` to get a CSV instead,
or ``--json`` for a JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


DEFAULT_A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"


class RPath:
    """Tiny Path-like wrapper for remote (Unix) paths used over SSH."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __str__(self) -> str:
        return self._path

    def __repr__(self) -> str:  # pragma: no cover
        return f"RPath({self._path!r})"

    def __fspath__(self) -> str:  # pragma: no cover
        return self._path

    @property
    def parent(self) -> "RPath":
        if "/" not in self._path:
            return RPath(".")
        return RPath(self._path.rsplit("/", 1)[0])

    @property
    def name(self) -> str:
        return self._path.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        n = self.name
        if "." in n:
            return n.rsplit(".", 1)[0]
        return n

    @property
    def suffix(self) -> str:
        n = self.name
        if "." in n:
            return "." + n.rsplit(".", 1)[-1]
        return ""

    def with_suffix(self, suffix: str) -> "RPath":
        return RPath(str(self.parent) + "/" + self.stem + suffix)


PathLike = Union[Path, RPath]


@dataclass
class RunResult:
    name: str = ""
    log_path: str = ""
    checkpoint_path: str = ""
    config_path: str = ""
    best_val_mpjpe: Optional[float] = None
    best_epoch: Optional[int] = None
    last_val_mpjpe: Optional[float] = None
    last_epoch: Optional[int] = None
    test_mpjpe: Optional[float] = None
    training_time_str: str = "N/A"
    notes: List[str] = field(default_factory=list)

    def val_str(self) -> str:
        if self.best_val_mpjpe is None:
            return "N/A"
        return f"{self.best_val_mpjpe:.2f}"

    def last_val_str(self) -> str:
        if self.last_val_mpjpe is None:
            return "N/A"
        return f"{self.last_val_mpjpe:.2f}"

    def test_str(self) -> str:
        if self.test_mpjpe is None:
            return "N/A"
        return f"{self.test_mpjpe:.2f}"

    def best_epoch_str(self) -> str:
        return str(self.best_epoch) if self.best_epoch is not None else "N/A"

    def last_epoch_str(self) -> str:
        return str(self.last_epoch) if self.last_epoch is not None else "N/A"


def _ssh_run(host: str, cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, cmd],
        text=True,
        stderr=subprocess.STDOUT,
        errors="ignore",
    )


def _normalize_repo_path(path: str) -> str:
    """Undo MSYS/Git-Bash path conversion for explicit Unix repo paths.

    Git Bash on Windows may convert ``/mnt/...`` arguments into
    ``C:/Program Files/Git/mnt/...``.  Detect the most common mangling and
    restore the original Unix path so that remote SSH commands still work.
    """
    if path.startswith("/mnt/"):
        return path
    # Common MSYS mangling: e.g. C:/Program Files/Git/mnt/... -> /mnt/...
    idx = path.lower().find("/mnt/")
    if idx != -1 and path[0].isalpha() and path[1] == ":":
        return path[idx:]
    return path


def _file_text(path: PathLike, ssh_host: Optional[str] = None) -> str:
    if ssh_host:
        return _ssh_run(ssh_host, f"cat '{path}'")
    return path.read_text(errors="ignore")  # type: ignore[union-attr]


def _file_exists(path: PathLike, ssh_host: Optional[str] = None) -> bool:
    if ssh_host:
        try:
            _ssh_run(ssh_host, f"test -f '{path}'")
            return True
        except subprocess.CalledProcessError:
            return False
    return path.exists()  # type: ignore[union-attr]


def _mtime(path: PathLike, ssh_host: Optional[str] = None) -> Optional[float]:
    """Return file mtime as a Unix timestamp, or None if unavailable."""
    if ssh_host:
        try:
            out = _ssh_run(ssh_host, f"stat -c %Y '{path}'").strip()
            return float(out)
        except (subprocess.CalledProcessError, ValueError):
            return None
    try:
        return path.stat().st_mtime  # type: ignore[union-attr]
    except OSError:
        return None


def _is_dir(path: PathLike, ssh_host: Optional[str]) -> bool:
    if ssh_host:
        try:
            out = _ssh_run(ssh_host, f"test -d '{path}' && echo yes || echo no")
            return out.strip() == "yes"
        except subprocess.CalledProcessError:
            return False
    return path.is_dir()  # type: ignore[union-attr]


def _latest_by_suffix(
    directory: PathLike, suffix: str, ssh_host: Optional[str]
) -> Optional[PathLike]:
    if ssh_host:
        try:
            out = _ssh_run(
                ssh_host,
                f"find '{directory}' -maxdepth 1 -type f -name '*{suffix}' -printf '%T@ %p\\n' "
                f"2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-",
            )
            found = out.strip()
            return RPath(found) if found else None
        except subprocess.CalledProcessError:
            return None
    files = sorted(Path(str(directory)).glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _find_sibling(
    path: PathLike, suffix: str, ssh_host: Optional[str]
) -> Optional[PathLike]:
    candidate = path.with_suffix(suffix)
    if _file_exists(candidate, ssh_host):
        return candidate
    # Some outputs use ``<stem>.config.json`` instead of ``<stem>.json``.
    if suffix == ".json":
        candidate = RPath(str(path.parent) + "/" + path.stem + ".config.json")
        if _file_exists(candidate, ssh_host):
            return candidate
    return None


def _resolve_input(
    raw: str,
    a800_repo: str,
    ssh_host: Optional[str],
) -> Tuple[str, Optional[PathLike], Optional[PathLike], Optional[PathLike]]:
    """Return (name, log, checkpoint, config) for a raw input string."""

    def _make_path(p: str) -> PathLike:
        return RPath(p) if ssh_host else Path(p)

    # Bare run name -> resolve under a800_repo/outputs/ablations.
    if "/" not in raw and "\\" not in raw:
        candidate_dir = _make_path(a800_repo + "/outputs/ablations/" + raw)
        candidate_log = _make_path(a800_repo + "/outputs/ablations/" + raw + ".log")

        # If a direct <run>.log exists, treat the run name as a file stem.
        if _file_exists(candidate_log, ssh_host):
            candidate = candidate_log
        elif _is_dir(candidate_dir, ssh_host):
            candidate = candidate_dir
        else:
            candidate = _make_path(a800_repo + "/outputs/ablations/" + raw)
    else:
        candidate = _make_path(raw)

    # Directory input.
    if _is_dir(candidate, ssh_host):
        log = _latest_by_suffix(candidate, ".log", ssh_host)
        ckpt = _latest_by_suffix(candidate, ".pth", ssh_host)
        cfg = _latest_by_suffix(candidate, ".config.json", ssh_host)
        name = raw.strip("/").split("/")[-1] or "unknown"
        return name, log, ckpt, cfg

    # File input.
    suffix = candidate.suffix
    if suffix == ".log":
        log: Optional[PathLike] = candidate
        ckpt = _find_sibling(candidate, ".pth", ssh_host)
        cfg = _find_sibling(candidate, ".config.json", ssh_host)
    elif suffix == ".pth":
        log = _find_sibling(candidate, ".log", ssh_host)
        ckpt = candidate
        cfg = _find_sibling(candidate, ".config.json", ssh_host)
    elif suffix == ".json" and candidate.name.endswith(".config.json"):
        log = _find_sibling(candidate, ".log", ssh_host)
        ckpt = _find_sibling(candidate, ".pth", ssh_host)
        cfg = candidate
    else:
        log = _find_sibling(candidate, ".log", ssh_host)
        ckpt = _find_sibling(candidate, ".pth", ssh_host)
        cfg = _find_sibling(candidate, ".config.json", ssh_host)

    name = candidate.stem
    if name.endswith(".config"):
        name = name[: -len(".config")]
    return name, log, ckpt, cfg


def _find_eval_json(
    log_path: Optional[PathLike],
    checkpoint_path: Optional[PathLike],
    ssh_host: Optional[str],
) -> Optional[PathLike]:
    """Look for an evaluation JSON file associated with a run."""
    candidates: List[PathLike] = []
    if checkpoint_path is not None:
        base = checkpoint_path.stem
        candidates.extend(
            [
                checkpoint_path.with_suffix(".eval.json"),
                RPath(str(checkpoint_path.parent) + f"/{base}_eval.json"),
                RPath(str(checkpoint_path.parent) + f"/{base}_test.json"),
                RPath(str(checkpoint_path.parent) + f"/eval_{base}.json"),
            ]
        )
    if log_path is not None:
        base = log_path.stem
        candidates.extend(
            [
                log_path.with_suffix(".eval.json"),
                RPath(str(log_path.parent) + f"/{base}_eval.json"),
                RPath(str(log_path.parent) + f"/{base}_test.json"),
                RPath(str(log_path.parent) + f"/eval_{base}.json"),
            ]
        )

    seen: set = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if _file_exists(c, ssh_host):
            return c
    return None


def _extract_val_series(text: str) -> List[Tuple[int, float]]:
    """Return (epoch, val_mpjpe_mm) sorted by epoch."""
    pattern = re.compile(
        r"Epoch\s+(\d+):\s+"
        r"train_loss=[\d.eE+-]+,\s+"
        r"val_loss=[\d.eE+-]+,\s+"
        r"val_MPJPE=([\d.]+)mm"
    )
    matches = pattern.findall(text)
    out: List[Tuple[int, float]] = []
    for epoch, val in matches:
        out.append((int(epoch), float(val)))
    return out


def _extract_test_from_text(text: str) -> Optional[float]:
    """Try to extract a test MPJPE value from free-form log text."""
    patterns = [
        r"MPJPE:\s*([\d.]+)\s*mm",
        r"test_MPJPE[=:\s]+([\d.]+)\s*mm?",
        r"test mpjpe[=:\s]+([\d.]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _extract_training_time(text: str) -> Optional[str]:
    """Extract a human-readable training time from log text if present."""
    patterns = [
        r"[Tt]otal\s+[Tt]raining\s+[Tt]ime\s*[:=]\s*([^\n]+)",
        r"[Tt]raining\s+[Tt]ime\s*[:=]\s*([^\n]+)",
        r"[Ee]lapsed\s+[Tt]ime\s*[:=]\s*([^\n]+)",
        r"time\s*[:=]\s*([\d]+m?[\d]+s[^\n]*)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].strip()
    return None


def _format_duration(seconds: float) -> str:
    """Convert seconds to a compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _extract_test_from_json(json_text: str) -> Optional[float]:
    """Extract MPJPE from an eval JSON string."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    for key in ("mpjpe", "MPJPE", "test_mpjpe", "test_MPJPE"):
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _parse_run(
    raw: str,
    a800_repo: str,
    ssh_host: Optional[str],
) -> RunResult:
    name, log_path, ckpt_path, cfg_path = _resolve_input(raw, a800_repo, ssh_host)
    result = RunResult(
        name=name,
        log_path=str(log_path) if log_path else "",
        checkpoint_path=str(ckpt_path) if ckpt_path else "",
        config_path=str(cfg_path) if cfg_path else "",
    )

    if log_path is None and ckpt_path is None and cfg_path is None:
        result.notes.append("no log/checkpoint/config found")
        return result

    text = ""
    if log_path is not None:
        try:
            text = _file_text(log_path, ssh_host)
        except subprocess.CalledProcessError:
            result.notes.append(f"could not read {log_path}")

    # Validation series.
    val_series = _extract_val_series(text)
    if val_series:
        best_epoch, best_val = min(val_series, key=lambda x: x[1])
        last_epoch, last_val = val_series[-1]
        result.best_val_mpjpe = best_val
        result.best_epoch = best_epoch
        result.last_val_mpjpe = last_val
        result.last_epoch = last_epoch
    else:
        result.notes.append("no val_MPJPE lines yet")

    # Test MPJPE.
    test_val = _extract_test_from_text(text)
    if test_val is not None:
        result.test_mpjpe = test_val
    else:
        eval_json = _find_eval_json(log_path, ckpt_path, ssh_host)
        if eval_json is not None:
            try:
                json_text = _file_text(eval_json, ssh_host)
                test_val = _extract_test_from_json(json_text)
                if test_val is not None:
                    result.test_mpjpe = test_val
            except subprocess.CalledProcessError:
                pass

    # Training time.
    time_str = _extract_training_time(text)
    if time_str:
        result.training_time_str = time_str
    else:
        log_mtime = _mtime(log_path, ssh_host) if log_path else None
        ckpt_mtime = _mtime(ckpt_path, ssh_host) if ckpt_path else None
        if log_mtime is not None and ckpt_mtime is not None and ckpt_mtime > log_mtime:
            result.training_time_str = _format_duration(ckpt_mtime - log_mtime)
        elif ckpt_mtime is not None:
            result.training_time_str = f"ckpt @{datetime.fromtimestamp(ckpt_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        elif log_mtime is not None:
            result.training_time_str = f"log @{datetime.fromtimestamp(log_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"

    return result


def _markdown_table(rows: List[RunResult]) -> str:
    header = ["Run", "Best val (mm)", "@epoch", "Last val (mm)", "@epoch", "Test (mm)", "Training time"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        cells = [
            r.name,
            r.val_str(),
            r.best_epoch_str(),
            r.last_val_str(),
            r.last_epoch_str(),
            r.test_str(),
            r.training_time_str,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _csv(rows: List[RunResult]) -> str:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run", "best_val_mpjpe_mm", "best_epoch", "last_val_mpjpe_mm", "last_epoch", "test_mpjpe_mm", "training_time", "log_path", "checkpoint_path"])
    for r in rows:
        writer.writerow(
            [
                r.name,
                r.best_val_mpjpe if r.best_val_mpjpe is not None else "",
                r.best_epoch if r.best_epoch is not None else "",
                r.last_val_mpjpe if r.last_val_mpjpe is not None else "",
                r.last_epoch if r.last_epoch is not None else "",
                r.test_mpjpe if r.test_mpjpe is not None else "",
                r.training_time_str,
                r.log_path,
                r.checkpoint_path,
            ]
        )
    return buf.getvalue()


def _json_output(rows: List[RunResult]) -> str:
    return json.dumps(
        [
            {
                "run": r.name,
                "best_val_mpjpe_mm": r.best_val_mpjpe,
                "best_epoch": r.best_epoch,
                "last_val_mpjpe_mm": r.last_val_mpjpe,
                "last_epoch": r.last_epoch,
                "test_mpjpe_mm": r.test_mpjpe,
                "training_time": r.training_time_str,
                "log_path": r.log_path,
                "checkpoint_path": r.checkpoint_path,
                "config_path": r.config_path,
                "notes": r.notes,
            }
            for r in rows
        ],
        indent=2,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare true-GT H36M ablations from A800/local training outputs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Log files, checkpoint files, config files, directory paths, or run names.",
    )
    parser.add_argument(
        "--a800-repo",
        default=DEFAULT_A800_REPO,
        help="Path to the A800 training repo (used to resolve bare run names).",
    )
    parser.add_argument("--ssh-host", default=None, help="SSH host alias for A800-D.")
    parser.add_argument("--csv", action="store_true", help="Output CSV instead of Markdown.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown.")
    parser.add_argument(
        "--sort-by",
        choices=["best_val", "last_val", "name"],
        default="best_val",
        help="Sort order for the output table.",
    )
    args = parser.parse_args(argv)

    # Undo MSYS/Git-Bash path conversion for explicit Unix repo paths.
    a800_repo = _normalize_repo_path(args.a800_repo)

    rows: List[RunResult] = []
    for raw in args.inputs:
        try:
            rows.append(_parse_run(raw, a800_repo, args.ssh_host))
        except subprocess.CalledProcessError as exc:
            err = RunResult(name=raw)
            err.notes.append(f"SSH/command error: {exc}")
            rows.append(err)
        except Exception as exc:
            err = RunResult(name=raw)
            err.notes.append(f"parse error: {exc}")
            rows.append(err)

    def sort_key(r: RunResult) -> Tuple[float, float, str]:
        if args.sort_by == "best_val":
            return (r.best_val_mpjpe if r.best_val_mpjpe is not None else float("inf"), 0.0, r.name)
        if args.sort_by == "last_val":
            return (r.last_val_mpjpe if r.last_val_mpjpe is not None else float("inf"), 0.0, r.name)
        return (0.0, 0.0, r.name)

    rows.sort(key=sort_key)

    if args.json:
        print(_json_output(rows))
    elif args.csv:
        print(_csv(rows))
    else:
        print(_markdown_table(rows))

    if any(r.notes for r in rows):
        print("\nNotes:", file=sys.stderr)
        for r in rows:
            if r.notes:
                print(f"  {r.name}: {', '.join(r.notes)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
