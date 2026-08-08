#!/usr/bin/env python3
"""Auto-eval + full-scale training launcher for OmniMultiViewFusion baselines.

Polls training logs in ``outputs/`` for ``val_MPJPE`` completion and, once a
baseline run is finished, triggers an evaluation pass and launches the matching
``*_fullscale.sh`` run in a detached tmux session.

The script is safe to run repeatedly: it writes a ``.auto_eval_done`` marker
next to each log and only re-evaluates when the log has been updated since the
marker.

Usage (daemon mode, poll every 60 s):
    python scripts/auto_eval_when_ready.py

Single-shot dry-run:
    python scripts/auto_eval_when_ready.py --once --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT / "outputs"
DEFAULT_LOG_GLOB = "omniview_fusion_v*.log"
DEFAULT_GPU = 0
DEFAULT_POLL_INTERVAL = 60
DEFAULT_OVERFIT_PATIENCE = 3
DEFAULT_OVERFIT_MIN_EPOCHS = 5
LOCK_FILE = Path("/tmp/motionflow_auto_eval_py.lock")


@dataclass(frozen=True)
class Experiment:
    name: str
    log_path: Path
    checkpoint: Path
    config: Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}")


def _run(cmd: List[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def find_candidate_logs(log_dir: Path, pattern: str) -> List[Path]:
    """Find training logs that are not full-scale logs."""
    logs = sorted(log_dir.glob(pattern))
    return [p for p in logs if "_fullscale" not in p.name]


def has_val_mpjpe_completed(log_path: Path) -> bool:
    """Return True if the log contains at least one val_MPJPE line."""
    try:
        with log_path.open("r", errors="ignore") as f:
            for line in f:
                if "val_MPJPE=" in line:
                    return True
    except FileNotFoundError:
        return False
    return False


def best_val_mpjpe(log_path: Path) -> Optional[float]:
    """Return the best (lowest) validation MPJPE in mm from the log."""
    try:
        text = log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return None
    vals = [float(m) for m in re.findall(r"val_MPJPE=([\d.]+)mm", text)]
    return min(vals) if vals else None


def last_val_mpjpe(log_path: Path) -> Optional[float]:
    """Return the last validation MPJPE in mm from the log."""
    try:
        text = log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return None
    matches = re.findall(r"val_MPJPE=([\d.]+)mm", text)
    return float(matches[-1]) if matches else None


def extract_val_mpjpe_series(log_path: Path) -> List[float]:
    """Return all validation MPJPE values in chronological order."""
    try:
        text = log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return []
    return [float(m) for m in re.findall(r"val_MPJPE=([\d.]+)mm", text)]


def best_epoch_info(log_path: Path) -> Tuple[Optional[int], Optional[float]]:
    """Return (epoch_index, best_val_mpjpe) from the log, 0-based.

    Returns (None, None) when no val_MPJPE lines are present.
    """
    series = extract_val_mpjpe_series(log_path)
    if not series:
        return None, None
    best_idx = min(range(len(series)), key=series.__getitem__)
    return best_idx, series[best_idx]


def detect_overfitting(
    log_path: Path,
    patience: int = DEFAULT_OVERFIT_PATIENCE,
    min_epochs: int = DEFAULT_OVERFIT_MIN_EPOCHS,
) -> Tuple[bool, int]:
    """Detect whether validation MPJPE is rising monotonically.

    Args:
        log_path: training log to analyse.
        patience: number of consecutive increasing val_MPJPE epochs required
                  to flag overfitting.
        min_epochs: require at least this many validation epochs before
                    overfit detection activates (avoids early noise).

    Returns:
        (is_overfitting, epochs_since_best)
    """
    series = extract_val_mpjpe_series(log_path)
    if len(series) < max(patience + 1, min_epochs):
        return False, 0

    best_idx, _ = best_epoch_info(log_path)
    if best_idx is None:
        return False, 0

    epochs_since_best = len(series) - 1 - best_idx

    # Count consecutive increases at the tail of the series.
    consecutive_increases = 0
    for i in range(len(series) - 1, 0, -1):
        if series[i] > series[i - 1]:
            consecutive_increases += 1
        else:
            break

    is_overfitting = consecutive_increases >= patience
    return is_overfitting, epochs_since_best


def map_to_fullscale_script(experiment_name: str) -> Optional[Path]:
    """Map an experiment name like 'omniview_fusion_v13_temporal' to its full-scale script."""
    script_dir = ROOT / "scripts"
    candidates = [
        script_dir / f"run_{experiment_name}_fullscale.sh",
        script_dir / f"run_{experiment_name}.sh",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def map_to_eval_script(experiment_name: str) -> Optional[Path]:
    """Map an experiment name to a matching eval script if one exists."""
    script_dir = ROOT / "scripts"
    short = experiment_name
    if short.startswith("omniview_fusion_"):
        short = short[len("omniview_fusion_"):]
    candidates = [
        script_dir / f"eval_{short}.sh",
        script_dir / f"eval_{short}_wsl.sh",
        script_dir / f"eval_omniview_fusion_v2_{short}.sh",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def discover_experiments(log_dir: Path, pattern: str) -> List[Experiment]:
    logs = find_candidate_logs(log_dir, pattern)
    experiments = []
    for log_path in logs:
        name = log_path.stem
        checkpoint = log_path.with_suffix(".pth")
        config = log_path.with_suffix(".config.json")
        experiments.append(Experiment(name=name, log_path=log_path, checkpoint=checkpoint, config=config))
    return experiments


def _marker_path(log_path: Path) -> Path:
    return log_path.with_suffix(log_path.suffix + ".auto_eval_done")


def is_ready(log_path: Path) -> bool:
    """Check whether the log has a completed val_MPJPE and a newer checkpoint."""
    if not has_val_mpjpe_completed(log_path):
        return False
    marker = _marker_path(log_path)
    if not marker.exists():
        return True
    return log_path.stat().st_mtime > marker.stat().st_mtime


def mark_done(log_path: Path) -> None:
    _marker_path(log_path).touch()


def run_eval(experiment: Experiment, eval_command: Optional[str], dry_run: bool) -> bool:
    """Trigger evaluation for the experiment. Returns True if eval succeeded or was skipped."""
    if eval_command:
        cmd = ["bash", "-c", eval_command]
    else:
        eval_script = map_to_eval_script(experiment.name)
        if eval_script is None:
            _log(f"[{experiment.name}] No eval script found; skipping eval. Pass --eval-command to override.")
            return True
        cmd = ["bash", str(eval_script)]

    _log(f"[{experiment.name}] Running eval: {' '.join(cmd)}")
    if dry_run:
        _log(f"[{experiment.name}] DRY-RUN: would run eval: {' '.join(cmd)}")
        return True

    try:
        proc = _run(cmd, cwd=ROOT, check=False)
        if proc.returncode != 0:
            _log(f"[{experiment.name}] Eval failed (exit={proc.returncode}):\n{proc.stderr}")
            return False
        _log(f"[{experiment.name}] Eval completed.")
        return True
    except Exception as exc:
        _log(f"[{experiment.name}] Eval exception: {exc}")
        return False


def run_fullscale(experiment: Experiment, gpu: int, dry_run: bool) -> bool:
    """Launch the matching full-scale run in a detached tmux session."""
    script = map_to_fullscale_script(experiment.name)
    if script is None:
        _log(f"[{experiment.name}] No fullscale script found; skipping launch.")
        return False

    session_name = f"{experiment.name}_fullscale_a800"
    cmd = [
        "tmux", "new-session", "-d", "-s", session_name, "-n", "main",
        f"source .venv/bin/activate && CUDA_VISIBLE_DEVICES={gpu} bash {script}",
    ]

    _log(f"[{experiment.name}] Launching fullscale run in tmux session '{session_name}' on GPU {gpu}: {script}")
    if dry_run:
        _log(f"[{experiment.name}] DRY-RUN: would launch {' '.join(cmd)}")
        return True

    try:
        _run(cmd, cwd=ROOT)
        _log(f"[{experiment.name}] Fullscale tmux session '{session_name}' started.")
        return True
    except subprocess.CalledProcessError as exc:
        _log(f"[{experiment.name}] Failed to start fullscale tmux session: {exc.stderr}")
        return False


def process_experiment(experiment: Experiment, args: argparse.Namespace) -> bool:
    """Process a single experiment; returns True if it was handled."""
    if not is_ready(experiment.log_path):
        return False

    best = best_val_mpjpe(experiment.log_path)
    last = last_val_mpjpe(experiment.log_path)
    is_overfitting, epochs_since_best = detect_overfitting(
        experiment.log_path,
        patience=args.overfit_patience,
        min_epochs=args.overfit_min_epochs,
    )
    status_msg = (
        f"Detected completed baseline (best={best}mm, last={last}mm, "
        f"epochs_since_best={epochs_since_best})"
    )
    if is_overfitting:
        status_msg += " — OVERFITTING DETECTED"
    _log(f"[{experiment.name}] {status_msg}")

    if not args.skip_eval:
        if not run_eval(experiment, args.eval_command, args.dry_run):
            return False

    if not args.skip_fullscale:
        if is_overfitting and not args.force_fullscale:
            _log(
                f"[{experiment.name}] Skipping full-scale launch due to overfitting. "
                f"Use --force-fullscale to override."
            )
        else:
            if not run_fullscale(experiment, args.gpu, args.dry_run):
                return False

    if not args.dry_run:
        mark_done(experiment.log_path)
    return True


def acquire_lock(lock_file: Path) -> bool:
    """Attempt to acquire the singleton lock; returns True on success."""
    try:
        if lock_file.exists():
            try:
                pid = int(lock_file.read_text().strip())
                try:
                    os.kill(pid, 0)
                    return False
                except (ProcessLookupError, ValueError):
                    lock_file.unlink()
            except ValueError:
                lock_file.unlink()
        lock_file.write_text(str(os.getpid()))
        return True
    except Exception as exc:
        _log(f"Lock acquisition failed: {exc}")
        return False


def release_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink()
    except FileNotFoundError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-evaluate completed baselines and launch full-scale runs."
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help=f"Directory containing training logs (default: {DEFAULT_LOG_DIR})")
    parser.add_argument("--log-glob", type=str, default=DEFAULT_LOG_GLOB,
                        help=f"Glob pattern for training logs (default: {DEFAULT_LOG_GLOB})")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help=f"Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL})")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                        help=f"GPU index for full-scale runs (default: {DEFAULT_GPU})")
    parser.add_argument("--eval-command", type=str, default=None,
                        help="Custom shell command to run for evaluation. Overrides script discovery.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation step.")
    parser.add_argument("--skip-fullscale", action="store_true", help="Skip the full-scale launch step.")
    parser.add_argument("--once", action="store_true", help="Run a single polling cycle and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions but do not launch eval or fullscale runs.")
    parser.add_argument("--json-summary", type=Path, default=None,
                        help="If given, write a JSON summary of the polling results to this path.")
    parser.add_argument("--lock-file", type=Path, default=LOCK_FILE,
                        help=f"Path to the singleton lock file (default: {LOCK_FILE})")
    parser.add_argument("--overfit-patience", type=int, default=DEFAULT_OVERFIT_PATIENCE,
                        help="Number of consecutive increasing val_MPJPE epochs required to flag overfitting.")
    parser.add_argument("--overfit-min-epochs", type=int, default=DEFAULT_OVERFIT_MIN_EPOCHS,
                        help="Minimum validation epochs before overfit detection activates.")
    parser.add_argument("--force-fullscale", action="store_true",
                        help="Launch full-scale run even if overfitting is detected.")
    return parser


def poll(args: argparse.Namespace) -> Dict[str, object]:
    """Run one polling cycle and return a summary dict."""
    experiments = discover_experiments(args.log_dir, args.log_glob)
    summary: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log_dir": str(args.log_dir),
        "scanned": len(experiments),
        "processed": [],
        "skipped": [],
    }

    for experiment in experiments:
        if is_ready(experiment.log_path):
            overfitting, _ = detect_overfitting(
                experiment.log_path,
                patience=args.overfit_patience,
                min_epochs=args.overfit_min_epochs,
            )
            summary["processed"].append({
                "name": experiment.name,
                "overfitting": overfitting,
            })
            process_experiment(experiment, args)
        else:
            summary["skipped"].append(experiment.name)

    if args.json_summary:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        args.json_summary.write_text(json.dumps(summary, indent=2))

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not acquire_lock(args.lock_file):
        _log("Another auto_eval_when_ready.py is already running; exiting.")
        return 0

    try:
        _log("Auto-eval monitor started. dry_run=%s once=%s" % (args.dry_run, args.once))
        while True:
            poll(args)
            if args.once:
                _log("Single-shot complete; exiting.")
                break
            time.sleep(args.poll_interval)
    finally:
        release_lock(args.lock_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
