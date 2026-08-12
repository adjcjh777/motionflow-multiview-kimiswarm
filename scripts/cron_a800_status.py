#!/usr/bin/env python3
"""Read-only A800-D status monitor for v85/v86 training runs.

This script is intended to be invoked from a cron job on the local WSL host.
It SSH's into a800-D, inspects GPU usage / tmux sessions / running processes,
extracts the latest val_MPJPE from the v85/v86 training logs, and appends a
timestamped summary to outputs/cron_a800_status.log.

It never writes to or modifies anything on A800.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"
LOG_FILE = Path(__file__).resolve().parents[1] / "outputs" / "cron_a800_status.log"

# Runs we are interested in monitoring.
# `pname` is matched against the remote process command line, so it should be
# specific enough to distinguish runs that share the same Python module.
RUNS = {
    "v85_random_view_dropout": {
        "log": "outputs/ablations/v85_random_view_dropout_medium_a800.log",
        # Match the Python training process, not the bash wrapper.
        "pname": "python.*v85_random_view_dropout_medium_a800",
    },
    "v86_no_count_embedding": {
        "log": "outputs/ablations/v86_no_count_embedding_medium_a800.log",
        # Match the Python training process, not the bash wrapper.
        "pname": "python.*v86_no_count_embedding_medium_a800",
    },
}


def ssh(cmd: str, timeout: int = 30) -> str:
    """Run a command on A800-D via SSH and return stdout."""
    try:
        return subprocess.check_output(
            [
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                SSH_HOST,
                cmd,
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        return f"<ssh-error: {exc.returncode}: {exc.output.strip()}>"
    except subprocess.TimeoutExpired:
        return "<ssh-timeout>"


def gpu_status() -> list[tuple[int, str, str, str]]:
    """Return [(index, util, used_MiB, free_MiB), ...] for all GPUs."""
    out = ssh(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free "
        "--format=csv,noheader,nounits"
    )
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append((int(parts[0]), parts[1], parts[2], parts[3]))
    return rows


def project_gpu_status() -> list[tuple[int, str, str, str]]:
    """GPU status filtered to the project GPUs (6 and 7)."""
    return [row for row in gpu_status() if row[0] in (6, 7)]


def tmux_sessions() -> list[tuple[str, str]]:
    """Return [(session_name, gpu_or_host), ...] for active tmux sessions."""
    out = ssh("tmux ls 2>/dev/null || true")
    sessions = []
    for line in out.strip().splitlines():
        match = re.match(r"^([^:]+):\s+\d+\s+windows?", line)
        if match:
            name = match.group(1).strip()
            gpu_match = re.search(r"_gpu(\d+)$", name)
            sessions.append((name, gpu_match.group(1) if gpu_match else "-"))
    return sessions


def running_pids_for(name: str) -> list[str]:
    """Return the parent PID of the process whose cmdline contains `name`.

    The main training process spawns DataLoader workers that share the same
    command line, so we report only the smallest PID as the parent.
    """
    out = ssh(f"pgrep -af '{name}' || true")
    pids = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) >= 2:
            pids.append(parts[0])
    if not pids:
        return []
    # Smallest PID is the parent trainer; workers have larger PIDs.
    return [min(pids, key=lambda p: int(p))]


def latest_log_metric(log_path: str, pattern: str) -> str | None:
    """Return the last match of `pattern` in the remote log file, or None."""
    out = ssh(f"tail -n 500 '{A800_REPO}/{log_path}' 2>/dev/null || true")
    matches = re.findall(pattern, out)
    if matches:
        return matches[-1]
    return None


def latest_val_mpjpe(log_path: str) -> str | None:
    return latest_log_metric(log_path, r"val_MPJPE=([\d.]+)mm")


def latest_epoch(log_path: str) -> str | None:
    return latest_log_metric(log_path, r"Epoch\s+(\d+)[\s/]")


def disk_usage() -> str:
    out = ssh("df -h /mnt/nvme0n1p1 | tail -n 1 | awk '{print $5, $4}'")
    parts = out.strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} used, {parts[1]} free"
    return out.strip()


def build_report() -> str:
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"=== A800-D status @ {now} ===")

    # GPU status (project GPUs only)
    lines.append("GPU status:")
    for idx, util, used, free in project_gpu_status():
        lines.append(f"  GPU {idx}: util={util}, mem_used={used} MiB, mem_free={free} MiB")

    # Disk usage on A800
    lines.append(f"A800 disk (/mnt/nvme0n1p1): {disk_usage()}")

    # tmux sessions
    lines.append("tmux sessions:")
    sessions = tmux_sessions()
    if sessions:
        for name, gpu in sessions:
            lines.append(f"  {name} (gpu={gpu})")
    else:
        lines.append("  none")

    # Per-run status
    lines.append("run status:")
    for run_name, info in RUNS.items():
        pids = running_pids_for(info["pname"])
        status = "running" if pids else "not running"
        lines.append(f"  {run_name}: {status} (pids={','.join(pids) if pids else 'none'})")
        val = latest_val_mpjpe(info["log"])
        epoch = latest_epoch(info["log"])
        if val:
            lines.append(f"    latest val_MPJPE = {val} mm")
        if epoch:
            lines.append(f"    latest epoch = {epoch}")

    lines.append(""  )  # blank line between entries
    return "\n".join(lines)


def main() -> int:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = build_report()
    except Exception as exc:  # noqa: BLE001
        report = f"=== A800-D status @ {datetime.now(timezone.utc).isoformat()} ===\n<monitor-error: {exc}>\n"

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(report + "\n")

    # Also emit a concise line to stdout/stderr for cron logging.
    print(f"[cron-a800] logged to {LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
