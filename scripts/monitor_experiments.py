"""Monitor local and A800 experiment logs and print a compact status table.

Example::

    python scripts/monitor_experiments.py
    python scripts/monitor_experiments.py --a800 --watch 60
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path


A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"


def _extract_values(text: str, pattern: str) -> list[float]:
    return [float(v) for v in re.findall(pattern, text)]


def _parse_log(text: str) -> dict:
    train_steps = _extract_values(text, r"train step\s+\d+:\s*loss=([0-9.]+)")
    val_mpjpe = _extract_values(text, r"val_MPJPE=([0-9.]+)mm")
    best_val = _extract_values(text, r"Best val MPJPE:\s*([0-9.]+)mm")
    return {
        "last_train_loss": train_steps[-1] if train_steps else None,
        "last_val_mpjpe": val_mpjpe[-1] if val_mpjpe else None,
        "best_val_mpjpe": min(val_mpjpe) if val_mpjpe else (best_val[-1] if best_val else None),
        "n_epochs": len(val_mpjpe),
    }


def _local_runs(outputs_dir: Path) -> list[dict]:
    rows = []
    for log_path in outputs_dir.glob("*.log"):
        info = _parse_log(log_path.read_text(errors="ignore"))
        if info["best_val_mpjpe"] is not None or info["last_train_loss"] is not None:
            rows.append({
                "run": log_path.stem,
                "location": "local",
                **info,
            })
    return rows


def _a800_runs() -> list[dict]:
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        SSH_HOST,
        f"find {A800_REPO}/outputs -maxdepth 1 -name '*.log' -print0 | xargs -0 grep -H 'val_MPJPE=' 2>/dev/null || true",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, errors="ignore")
    except subprocess.CalledProcessError:
        return []

    by_run: dict[str, dict] = {}
    for line in out.splitlines():
        if "val_MPJPE=" not in line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        log_path = Path(parts[0].strip())
        rest = ":".join(parts[1:])
        run_name = log_path.stem
        val_mpjpe = _extract_values(rest, r"val_MPJPE=([0-9.]+)mm")
        if not val_mpjpe:
            continue
        if run_name not in by_run:
            by_run[run_name] = {
                "run": run_name,
                "location": "a800",
                "last_train_loss": None,
                "last_val_mpjpe": val_mpjpe[-1],
                "best_val_mpjpe": min(val_mpjpe),
                "n_epochs": len(val_mpjpe),
            }
        else:
            by_run[run_name]["last_val_mpjpe"] = val_mpjpe[-1]
            by_run[run_name]["best_val_mpjpe"] = min(by_run[run_name]["best_val_mpjpe"], *val_mpjpe)
            by_run[run_name]["n_epochs"] = max(by_run[run_name]["n_epochs"], len(val_mpjpe))
    return list(by_run.values())


def _fmt(x) -> str:
    if x is None:
        return "--"
    return f"{x:.2f}"


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No runs found.")
        return
    rows_sorted = sorted(rows, key=lambda r: r.get("best_val_mpjpe") or float("inf"))
    print(f"{'Run':<60} {'Loc':<6} {'Ep':>3} {'Best':>7} {'Latest':>7} {'TrainLoss':>10}")
    print("-" * 95)
    for row in rows_sorted:
        print(
            f"{row['run']:<60} "
            f"{row['location']:<6} "
            f"{row['n_epochs']:>3} "
            f"{_fmt(row['best_val_mpjpe']):>7} "
            f"{_fmt(row['last_val_mpjpe']):>7} "
            f"{_fmt(row['last_train_loss']):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor local and A800 experiment logs.")
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"), help="Local outputs directory")
    parser.add_argument("--a800", action="store_true", help="Also fetch A800 runs via SSH")
    parser.add_argument("--watch", type=int, default=0, help="If >0, refresh every N seconds")
    args = parser.parse_args()

    while True:
        rows = _local_runs(args.outputs_dir)
        if args.a800:
            rows.extend(_a800_runs())
        _print_table(rows)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
