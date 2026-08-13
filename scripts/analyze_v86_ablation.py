#!/usr/bin/env python3
"""Analyze the v86 no-count-embedding ablation and compare it with v85 and v25.

The script is read-only: it parses training logs/checkpoints and evaluation JSONs,
producing a Markdown report.  It never writes to the A800 repo; run it locally and
point it at the A800 repo (or a local mirror) and it will inspect files via SSH if
necessary.

What it reports
---------------
* Best validation MPJPE and the epoch it was reached.
* Early-stop / final epoch.
* Test-set MPJPE/PA-MPJPE for S9 and S11 (if ``eval_*.json`` exists).
* Sparse-view MPJPE@k for k=2,3,4 (if variable-view JSON exists).
* Delta tables comparing v86 vs v85 (count-embedding effect) and v86 vs v25.

Usage
-----
    # Local repo (run after files have been synced to local)
    python scripts/analyze_v86_ablation.py --repo . --out outputs/v86_ablation_analysis.md

    # Read from A800 over SSH (read-only)
    python scripts/analyze_v86_ablation.py \
        --repo a800-D:/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20 \
        --out outputs/v86_ablation_analysis.md

    # Dry-run / quiet
    python scripts/analyze_v86_ablation.py --repo . --quiet

The script handles missing files gracefully: sections that depend on files not yet
produced are marked as *pending*.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union


DEFAULT_RUNS = {
    "v86": "v86_no_count_embedding_medium_a800",
    "v85": "v85_random_view_dropout_medium_a800",
    "v25": "v25_true_gt_v2_medium_a800",
}

PathLike = Union[str, Path, PurePosixPath]


# -----------------------------------------------------------------------------
# Remote helpers
# -----------------------------------------------------------------------------


class RPath:
    """Tiny wrapper for remote (Unix) paths read over SSH."""

    def __init__(self, path: str):
        self._path = path

    def __str__(self) -> str:
        return self._path

    def __repr__(self) -> str:  # pragma: no cover
        return f"RPath({self._path!r})"

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


# -----------------------------------------------------------------------------
# Repo location abstraction
# -----------------------------------------------------------------------------


class RepoLocation:
    """Represents a repository on the local filesystem or an SSH host."""

    def __init__(self, spec: str):
        if ":" in spec and not spec.startswith("/"):
            # ssh-host:/path/to/repo
            self.ssh_host, _, self.root = spec.partition(":")
        else:
            self.ssh_host = None
            self.root = spec
        self._root_path = Path(self.root) if self.ssh_host is None else RPath(self.root)

    def is_local(self) -> bool:
        return self.ssh_host is None

    def _path(self, *parts: str) -> PathLike:
        p = str(self._root_path) + "/" + "/".join(parts)
        return Path(p) if self.is_local() else RPath(p)

    def read_text(self, path: PathLike) -> str:
        if self.ssh_host:
            return subprocess.check_output(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", self.ssh_host, f"cat '{path}'"],
                text=True,
                stderr=subprocess.STDOUT,
                errors="ignore",
            )
        return Path(path).read_text(errors="ignore")

    def exists(self, path: PathLike) -> bool:
        if self.ssh_host:
            try:
                subprocess.check_output(
                    ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", self.ssh_host, f"test -f '{path}'"],
                    stderr=subprocess.STDOUT,
                )
                return True
            except subprocess.CalledProcessError:
                return False
        return Path(path).exists()

    def list_files(self, directory: PathLike, pattern: str) -> List[str]:
        if self.ssh_host:
            try:
                out = subprocess.check_output(
                    ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", self.ssh_host,
                     f"find '{directory}' -maxdepth 1 -type f -name '{pattern}' 2>/dev/null"],
                    text=True,
                    stderr=subprocess.STDOUT,
                    errors="ignore",
                )
                return out.strip().splitlines()
            except subprocess.CalledProcessError:
                return []
        return [str(p) for p in Path(directory).glob(pattern)]


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------


@dataclass
class TrainingSummary:
    run_name: str
    best_val_mpjpe: Optional[float] = None
    best_epoch: Optional[int] = None
    final_val_mpjpe: Optional[float] = None
    final_epoch: Optional[int] = None
    early_stop_epoch: Optional[int] = None
    config_flags: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    S9_mpjpe: Optional[float] = None
    S9_pa: Optional[float] = None
    S11_mpjpe: Optional[float] = None
    S11_pa: Optional[float] = None
    path: str = ""


@dataclass
class VariableViewResult:
    per_dataset: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    path: str = ""

    def get(self, dataset: str, k: int) -> Optional[Dict[str, float]]:
        return self.per_dataset.get(dataset, {}).get(k)


def _parse_val_mpjpe_series(log_text: str) -> List[Tuple[int, float]]:
    pattern = re.compile(
        r"Epoch\s+(\d+):\s+train_loss=[\d.eE+-]+,\s+val_loss=[\d.eE+-]+,\s+val_MPJPE=([\d.]+)mm"
    )
    matches = pattern.findall(log_text)
    return [(int(epoch), float(val)) for epoch, val in matches]


def _extract_early_stop_epoch(log_text: str) -> Optional[int]:
    # e.g. "Early stopping at epoch 6 (no val_loss improvement for 3 epochs)."
    match = re.search(r"[Ee]arly\s+(?:stopping|stopped)\s+(?:at\s+)?epoch\s+(\d+)", log_text)
    if match:
        return int(match.group(1))
    return None


def _extract_best_val_line(log_text: str) -> Optional[Tuple[int, float]]:
    # e.g. "Best val MPJPE: 31.41mm -> outputs/ablations/..."
    match = re.search(r"[Bb]est\s+val\s+MPJPE:\s+([\d.]+)mm", log_text)
    if match:
        return float(match.group(1))
    return None


def parse_training_log(run_name: str, log_text: str) -> TrainingSummary:
    summary = TrainingSummary(run_name=run_name)
    series = _parse_val_mpjpe_series(log_text)
    if not series:
        summary.notes.append("No val_MPJPE lines found.")
        return summary

    summary.final_epoch, summary.final_val_mpjpe = series[-1]
    summary.early_stop_epoch = _extract_early_stop_epoch(log_text)

    # Best is the minimum val_MPJPE in the logged series.
    best_epoch, best_val = min(series, key=lambda x: x[1])
    summary.best_epoch = best_epoch
    summary.best_val_mpjpe = best_val

    # If the "Best val MPJPE" line exists, prefer it (it matches the saved ckpt).
    best_line = _extract_best_val_line(log_text)
    if best_line is not None:
        # Keep the epoch from the series; the line only gives the value.
        summary.best_val_mpjpe = best_line

    return summary


def load_config(repo: RepoLocation, path: PathLike) -> Dict[str, Any]:
    try:
        text = repo.read_text(path)
        return json.loads(text)
    except Exception as exc:  # pragma: no cover
        return {"_error": str(exc)}


def load_test_json(repo: RepoLocation, path: PathLike) -> Optional[TestResult]:
    if not repo.exists(path):
        return None
    try:
        data = json.loads(repo.read_text(path))
    except Exception:
        return None

    result = TestResult(path=str(path))
    if "S9" in data and "S11" in data:
        result.S9_mpjpe = _float_or_none(data["S9"], "mpjpe_mm")
        result.S9_pa = _float_or_none(data["S9"], "pa_mpjpe_mm")
        result.S11_mpjpe = _float_or_none(data["S11"], "mpjpe_mm")
        result.S11_pa = _float_or_none(data["S11"], "pa_mpjpe_mm")
    elif isinstance(data, dict):
        # Fallback for flat eval JSONs
        result.S9_mpjpe = _float_or_none(data, "mpjpe")
        result.S9_pa = _float_or_none(data, "pa_mpjpe")
    return result


def _float_or_none(d: Dict[str, Any], key: str) -> Optional[float]:
    val = d.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_variable_view_json(repo: RepoLocation, path: PathLike) -> Optional[VariableViewResult]:
    if not repo.exists(path):
        return None
    try:
        data = json.loads(repo.read_text(path))
    except Exception:
        return None

    result = VariableViewResult(path=str(path))
    per_dataset = data.get("per_dataset", {})
    for dataset, k_map in per_dataset.items():
        result.per_dataset[dataset] = {}
        for k_str, metrics in k_map.items():
            try:
                k = int(k_str)
            except (TypeError, ValueError):
                continue
            result.per_dataset[dataset][k] = {
                "mpjpe_at_k": float(metrics.get("mpjpe_at_k", metrics.get("mean_mm", 0.0))),
                "std_mm": float(metrics.get("std_mm", 0.0)),
                "n_subsets": int(metrics.get("n_subsets", 1)),
                "temporal_jerk": float(metrics.get("temporal_jerk", 0.0)),
            }
    return result


def merge_variable_view_results(results: List[VariableViewResult]) -> VariableViewResult:
    """Merge per-k JSON files into a single result."""
    merged = VariableViewResult()
    for r in results:
        for dataset, k_map in r.per_dataset.items():
            merged.per_dataset.setdefault(dataset, {}).update(k_map)
    return merged


# -----------------------------------------------------------------------------
# Run discovery
# -----------------------------------------------------------------------------


@dataclass
class RunBundle:
    name: str
    log: Optional[PathLike] = None
    checkpoint: Optional[PathLike] = None
    config: Optional[PathLike] = None
    test_json: Optional[PathLike] = None
    variable_view_json: Optional[PathLike] = None
    variable_view_per_k: List[PathLike] = field(default_factory=list)

    def training_summary(self, repo: RepoLocation) -> TrainingSummary:
        if self.log is None or not repo.exists(self.log):
            return TrainingSummary(run_name=self.name, notes=["Training log not found."])
        return parse_training_log(self.name, repo.read_text(self.log))


@dataclass
class Analysis:
    repo: RepoLocation
    v86: RunBundle
    v85: RunBundle
    v25: RunBundle


def _resolve_run_bundle(repo: RepoLocation, key: str, run_name: str) -> RunBundle:
    bundle = RunBundle(name=run_name)

    # Training artifacts
    log_path = repo._path("outputs", "ablations", f"{run_name}.log")
    ckpt_path = repo._path("outputs", "ablations", f"{run_name}.pth")
    cfg_path = repo._path("outputs", "ablations", f"{run_name}.config.json")

    bundle.log = log_path if repo.exists(log_path) else None
    bundle.checkpoint = ckpt_path if repo.exists(ckpt_path) else None
    bundle.config = cfg_path if repo.exists(cfg_path) else None

    # Test-set eval JSON (heuristic naming)
    test_candidates = [
        repo._path("outputs", f"eval_{run_name}_h36m_test.json"),
        repo._path("outputs", f"eval_{run_name}_h36m_test_a800.json"),
        repo._path("outputs", f"eval_{run_name}.json"),
    ]
    for c in test_candidates:
        if repo.exists(c):
            bundle.test_json = c
            break

    # Variable-view JSON and per-k files
    vv_candidates = [
        repo._path("outputs", f"variable_view_{run_name}.json"),
        repo._path("outputs", "variable_view_fix", f"variable_view_{run_name}.json"),
        repo._path("outputs", "variable_view_fix", f"variable_view_{run_name}_dlt_fallback.json"),
        repo._path("outputs", f"variable_view_{run_name}_dlt_fallback.json"),
    ]
    for c in vv_candidates:
        if repo.exists(c):
            bundle.variable_view_json = c
            break

    # Per-k files (either in outputs/ or outputs/variable_view_fix/)
    per_k_dir_parts = [
        ("outputs",),
        ("outputs", "variable_view_fix"),
    ]
    per_k_files: List[PathLike] = []
    for parts in per_k_dir_parts:
        for k in (2, 3, 4):
            p = repo._path(*parts, f"variable_view_{run_name}_k{k}.json")
            if repo.exists(p):
                per_k_files.append(p)
            p_fb = repo._path(*parts, f"variable_view_{run_name}_dlt_fallback_k{k}.json")
            if repo.exists(p_fb):
                per_k_files.append(p_fb)
    bundle.variable_view_per_k = per_k_files

    return bundle


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------


def _fmt(val: Optional[float], digits: int = 2) -> str:
    if val is None:
        return "*pending*"
    return f"{val:.{digits}f}"


def _delta(new: Optional[float], old: Optional[float]) -> str:
    if new is None or old is None:
        return "*pending*"
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def _pct(new: Optional[float], old: Optional[float]) -> str:
    if new is None or old is None or old == 0:
        return "*pending*"
    p = (new - old) / old * 100.0
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def _build_val_table(v86: TrainingSummary, v85: TrainingSummary, v25: TrainingSummary) -> str:
    rows = []
    for run, summary in [("v86", v86), ("v85", v85), ("v25", v25)]:
        rows.append(
            f"| {run} | {_fmt(summary.best_val_mpjpe)} | {summary.best_epoch if summary.best_epoch is not None else '*pending*'} | "
            f"{_fmt(summary.final_val_mpjpe)} | {summary.final_epoch if summary.final_epoch is not None else '*pending*'} | "
            f"{summary.early_stop_epoch if summary.early_stop_epoch is not None else '*pending*'} |"
        )
    return "\n".join(rows)


def _build_test_table(v86: Optional[TestResult], v85: Optional[TestResult], v25: Optional[TestResult]) -> str:
    rows = []
    for key, a, b in [("v86", v86, None), ("v85", v85, None), ("v25", v25, None)]:
        if a is None:
            rows.append(f"| {key} | *pending* | *pending* | *pending* | *pending* | *pending* |")
            continue
        rows.append(
            f"| {key} | {_fmt(a.S9_mpjpe)} | {_fmt(a.S9_pa)} | {_fmt(a.S11_mpjpe)} | {_fmt(a.S11_pa)} | "
            f"{_fmt(_weighted_test(a))} |"
        )
    return "\n".join(rows)


def _weighted_test(t: TestResult) -> Optional[float]:
    if t.S9_mpjpe is None or t.S11_mpjpe is None:
        return None
    # Weight by the frame counts used in official reports.
    # We do not have frame counts here; use a simple average.
    return (t.S9_mpjpe + t.S11_mpjpe) / 2.0


def _build_variable_view_table(vv86: VariableViewResult, vv85: VariableViewResult, vv25: VariableViewResult) -> str:
    lines: List[str] = []
    datasets_found = set(vv86.per_dataset.keys()) | set(vv85.per_dataset.keys()) | set(vv25.per_dataset.keys())
    datasets = [d for d in ("S9", "S11") if d in datasets_found]

    if not datasets:
        return "No variable-view results found for any run.\n"

    for dataset in datasets:
        lines.append(f"\n### {dataset}\n")
        lines.append("| k | v86 (mm) | v85 (mm) | v25 (mm) | Δ v86-v85 (mm) | Δ v86-v25 (mm) |")
        lines.append("|---|----------|----------|----------|----------------|----------------|")
        for k in (2, 3, 4):
            v86_metrics = vv86.get(dataset, k)
            v85_metrics = vv85.get(dataset, k)
            v25_metrics = vv25.get(dataset, k)
            v86_val = v86_metrics.get("mpjpe_at_k") if v86_metrics is not None else None
            v85_val = v85_metrics.get("mpjpe_at_k") if v85_metrics is not None else None
            v25_val = v25_metrics.get("mpjpe_at_k") if v25_metrics is not None else None
            lines.append(
                f"| {k} | {_fmt(v86_val)} | {_fmt(v85_val)} | {_fmt(v25_val)} | "
                f"{_delta(v86_val, v85_val)} | {_delta(v86_val, v25_val)} |"
            )

    return "\n".join(lines)


def _config_flag_line(config: Dict[str, Any], flag: str, default: Any = "*not set*") -> str:
    val = config.get(flag, default)
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def build_report(analysis: Analysis) -> str:
    v86_summary = analysis.v86.training_summary(analysis.repo)
    v85_summary = analysis.v85.training_summary(analysis.repo)
    v25_summary = analysis.v25.training_summary(analysis.repo)

    # Config flags relevant to the ablation
    v86_config = load_config(analysis.repo, analysis.v86.config) if analysis.v86.config else {}
    v85_config = load_config(analysis.repo, analysis.v85.config) if analysis.v85.config else {}

    # Test results
    v86_test = load_test_json(analysis.repo, analysis.v86.test_json) if analysis.v86.test_json else None
    v85_test = load_test_json(analysis.repo, analysis.v85.test_json) if analysis.v85.test_json else None
    v25_test = load_test_json(analysis.repo, analysis.v25.test_json) if analysis.v25.test_json else None

    # Variable-view results
    def _load_vv(bundle: RunBundle) -> VariableViewResult:
        results: List[VariableViewResult] = []
        if bundle.variable_view_json:
            r = load_variable_view_json(analysis.repo, bundle.variable_view_json)
            if r is not None:
                results.append(r)
        for p in bundle.variable_view_per_k:
            r = load_variable_view_json(analysis.repo, p)
            if r is not None:
                results.append(r)
        if len(results) == 1:
            return results[0]
        return merge_variable_view_results(results)

    vv86 = _load_vv(analysis.v86)
    vv85 = _load_vv(analysis.v85)
    vv25 = _load_vv(analysis.v25)

    report = f"""# v86 No-Count-Embedding Ablation Analysis

> Generated automatically by ``scripts/analyze_v86_ablation.py``.
> This report is read-only; it compares training and evaluation artifacts that already exist.

## 1. Training Summary

| Run | Best val MPJPE (mm) | @epoch | Final val MPJPE (mm) | @epoch | Early-stop epoch |
|-----|----------------------|--------|----------------------|--------|------------------|
{_build_val_table(v86_summary, v85_summary, v25_summary)}

### 1.1 Key config differences

| Flag | v86 | v85 |
|------|-----|-----|
| `use_random_view_dropout_v85` | {_config_flag_line(v86_config, "use_random_view_dropout_v85")} | {_config_flag_line(v85_config, "use_random_view_dropout_v85")} |
| `v85_dropout_prob` | {_config_flag_line(v86_config, "v85_dropout_prob")} | {_config_flag_line(v85_config, "v85_dropout_prob")} |
| `v85_min_views` | {_config_flag_line(v86_config, "v85_min_views")} | {_config_flag_line(v85_config, "v85_min_views")} |
| `v85_use_count_embedding` | {_config_flag_line(v86_config, "v85_use_count_embedding")} | {_config_flag_line(v85_config, "v85_use_count_embedding")} |

*The only deliberate difference should be `v85_use_count_embedding` (true for v85, false for v86).*

## 2. Test-Set Results (S9 / S11)

| Run | S9 MPJPE (mm) | S9 PA-MPJPE (mm) | S11 MPJPE (mm) | S11 PA-MPJPE (mm) | Avg (mm) |
|-----|---------------|------------------|----------------|-------------------|----------|
{_build_test_table(v86_test, v85_test, v25_test)}

## 3. Sparse-View Variable-View Results (MPJPE@k)

{_build_variable_view_table(vv86, vv85, vv25)}

## 4. Interpretation

* **Count-embedding contribution**: compare v86 vs v85. If v86 is worse at k<4 while full-view (k=4) is similar, the active-view-count embedding helps sparse-view generalization. If both are similar, dropout alone is sufficient and the embedding is non-essential.
* **Sparse-view gap**: compare both v85/v86 against the v25 DLT-fallback baseline (or any v25 variable-view numbers). Learned sparse-view results should eventually beat pure DLT fallback (~58/49 mm for k=2, ~33/25 mm for k=3 on S9/S11).
* **Full-view quality**: k=4 numbers indicate whether random view dropout harms the full-view learned model.

## 5. Raw artifact paths

| Run | Log | Checkpoint | Config |
|-----|-----|------------|--------|
| v86 | `{analysis.v86.log}` | `{analysis.v86.checkpoint}` | `{analysis.v86.config}` |
| v85 | `{analysis.v85.log}` | `{analysis.v85.checkpoint}` | `{analysis.v85.config}` |
| v25 | `{analysis.v25.log}` | `{analysis.v25.checkpoint}` | `{analysis.v25.config}` |

### Test / variable-view files found

| Run | Test JSON | Variable-view JSON | Per-k files |
|-----|-----------|--------------------|-------------|
| v86 | `{analysis.v86.test_json}` | `{analysis.v86.variable_view_json}` | {len(analysis.v86.variable_view_per_k)} |
| v85 | `{analysis.v85.test_json}` | `{analysis.v85.variable_view_json}` | {len(analysis.v85.variable_view_per_k)} |
| v25 | `{analysis.v25.test_json}` | `{analysis.v25.variable_view_json}` | {len(analysis.v25.variable_view_per_k)} |
"""
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the v86 no-count-embedding ablation vs v85 and v25."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Path to the project repo. Use ssh-host:/path syntax to read from A800 (read-only).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/v86_ablation_analysis.md",
        help="Output Markdown report path. Default: outputs/v86_ablation_analysis.md",
    )
    parser.add_argument(
        "--v86-run",
        type=str,
        default=DEFAULT_RUNS["v86"],
        help="Run name for v86 ablation.",
    )
    parser.add_argument(
        "--v85-run",
        type=str,
        default=DEFAULT_RUNS["v85"],
        help="Run name for v85 ablation.",
    )
    parser.add_argument(
        "--v25-run",
        type=str,
        default=DEFAULT_RUNS["v25"],
        help="Run name for v25 baseline.",
    )
    parser.add_argument(
        "--v25-var-view-run",
        type=str,
        default=None,
        help="Optional separate run name for v25 variable-view results (e.g. v25_true_gt_stability_a800_dlt_fallback).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the report to stdout, only write the file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo = RepoLocation(args.repo)
    v25_bundle = _resolve_run_bundle(repo, "v25", args.v25_run)
    if args.v25_var_view_run:
        vv_bundle = _resolve_run_bundle(repo, "v25", args.v25_var_view_run)
        v25_bundle.variable_view_json = vv_bundle.variable_view_json
        v25_bundle.variable_view_per_k = vv_bundle.variable_view_per_k
    analysis = Analysis(
        repo=repo,
        v86=_resolve_run_bundle(repo, "v86", args.v86_run),
        v85=_resolve_run_bundle(repo, "v85", args.v85_run),
        v25=v25_bundle,
    )

    report = build_report(analysis)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    if not args.quiet:
        print(report)

    print(f"\nReport written to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
