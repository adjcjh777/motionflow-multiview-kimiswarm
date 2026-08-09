#!/usr/bin/env python3
"""Quick A800-D status dashboard for the v31/v32/v33/v34/v35/v36 queue."""

from __future__ import annotations

import re
import subprocess
import sys


A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"


def a800_ssh(cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def gpu_status() -> list[tuple[int, int, int]]:
    out = a800_ssh("nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits")
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def tmux_sessions() -> list[tuple[str, int]]:
    try:
        out = a800_ssh("tmux ls 2>/dev/null || true")
    except subprocess.CalledProcessError:
        return []
    sessions = []
    for line in out.splitlines():
        # Match session names ending with _gpuN
        match = re.search(r"^([\w_]+)_gpu(\d+):", line)
        if match:
            sessions.append((match.group(1), int(match.group(2))))
    return sessions


def latest_val_mpjpe(session: str) -> str:
    """Return the latest val_MPJPE from any matching run log if available."""
    # Try to find a log file whose name contains the session run key.
    try:
        # Session name is like <run_key>_gpu<N>.  Look for any .log in outputs
        # that contains the run key.
        run_key = session.rsplit("_gpu", 1)[0]
        files = a800_ssh(
            rf"ls {A800_REPO}/outputs/ | grep -E '\.log$' | grep '{run_key}' || true"
        )
        log_names = [f.strip() for f in files.strip().splitlines() if f.strip()]
        if not log_names:
            return "N/A"
        # Prefer the most recently modified matching log.
        log_path = f"{A800_REPO}/outputs/{log_names[0]}"
        out = a800_ssh(f"tail -n 200 '{log_path}' 2>/dev/null || true")
    except (subprocess.CalledProcessError, IndexError):
        return "N/A"
    matches = re.findall(r"val_MPJPE=([\d.]+)mm", out)
    if matches:
        return f"{matches[-1]} mm"
    return "N/A"


def main() -> None:
    print("A800-D GPU status")
    print("-" * 40)
    for idx, used, free in gpu_status():
        print(f"  GPU {idx}: {used:6d} MiB used, {free:6d} MiB free")

    print("\nA800-D tmux sessions")
    print("-" * 40)
    sessions = tmux_sessions()
    if not sessions:
        print("  none")
        return

    # Group by GPU
    sessions_by_gpu: dict[int, list[str]] = {}
    for name, gpu in sessions:
        sessions_by_gpu.setdefault(gpu, []).append(name)

    for gpu in sorted(sessions_by_gpu):
        print(f"  GPU {gpu}:")
        for name in sessions_by_gpu[gpu]:
            val = latest_val_mpjpe(name)
            print(f"    {name:60s} val_MPJPE={val}")


if __name__ == "__main__":
    main()
