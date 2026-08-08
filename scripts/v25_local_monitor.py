#!/usr/bin/env python
"""Real-time monitor for the local RTX 4090 v25 small baseline.

Parses the training log written by ``experiments/train_omniview_fusion_v5_webbridge_multi.py``
and writes a rolling CSV of epoch-level metrics plus a small JSON status file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sys
import time
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+):\s+"
    r"train_loss=([\d.]+(?:e[+-]?\d+)?),\s+"
    r"val_loss=([\d.]+(?:e[+-]?\d+)?),\s+"
    r"val_MPJPE=([\d.]+)mm"
    r"(?:\s+\(saved\))?"
)


def parse_log(log_path: Path) -> list[dict]:
    """Return a list of epoch records from *log_path*."""
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="ignore")
    records: list[dict] = []
    for m in EPOCH_RE.finditer(text):
        records.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)),
                "val_mpjpe_mm": float(m.group(4)),
                "saved": "(saved)" in m.group(0),
            }
        )
    return records


def read_existing_epochs(csv_path: Path) -> set[int]:
    """Return the set of epochs already present in the CSV."""
    if not csv_path.exists():
        return set()
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return {int(row["epoch"]) for row in reader if row.get("epoch")}
    except Exception:
        return set()


def append_records(records: list[dict], csv_path: Path) -> None:
    """Append only new epoch records to the CSV."""
    if not records:
        return
    existing = read_existing_epochs(csv_path)
    new = [r for r in records if r["epoch"] not in existing]
    if not new:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for r in new:
        r["timestamp"] = now
    fieldnames = ["timestamp", "epoch", "train_loss", "val_loss", "val_mpjpe_mm", "saved"]
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new)
    else:
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(new)


def write_status(records: list[dict], status_path: Path, running: bool) -> None:
    """Write a small JSON status blob summarising the run so far."""
    best: dict | None = None
    if records:
        best_rec = min(records, key=lambda r: r["val_mpjpe_mm"])
        best = {"epoch": best_rec["epoch"], "val_mpjpe_mm": best_rec["val_mpjpe_mm"]}
    last = records[-1] if records else None
    status = {
        "status": "running" if running else "completed",
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_epochs_recorded": len(records),
        "best": best,
        "last": last,
    }
    status_path.write_text(json.dumps(status, indent=2))


def is_process_alive(pid: int) -> bool:
    """Return True if process *pid* is still running."""
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(1, False, pid)
        if handle:
            kernel.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor a v25 local 4090 training log.")
    parser.add_argument("--log", required=True, type=Path, help="Path to the training log.")
    parser.add_argument("--csv", default="outputs/v25_local_4090_monitor.csv", type=Path, help="Output CSV.")
    parser.add_argument("--status", default="outputs/v25_local_4090_status.json", type=Path, help="Output status JSON.")
    parser.add_argument("--pid", type=int, default=None, help="Training PID to watch.")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between log polls.")
    parser.add_argument("--stale-epochs", type=int, default=3, help="Exit after this many epochs with no new records once training PID is gone.")
    args = parser.parse_args(argv)

    def _handle_sig(signum, frame):
        write_status(parse_log(args.log), args.status, running=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    epochs_without_new = 0
    last_count = 0
    while True:
        records = parse_log(args.log)
        append_records(records, args.csv)

        running = args.pid is None or is_process_alive(args.pid)
        write_status(records, args.status, running)

        if len(records) > last_count:
            epochs_without_new = 0
            last_count = len(records)
        else:
            epochs_without_new += 1

        if not running and epochs_without_new >= args.stale_epochs:
            break
        if running:
            epochs_without_new = 0

        time.sleep(args.poll_interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
