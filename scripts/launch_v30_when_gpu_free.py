#!/usr/bin/env python3
"""Poll A800-D GPU memory and launch v30a when enough memory frees.

Usage:
    python scripts/launch_v30_when_gpu_free.py
"""

from __future__ import annotations

import subprocess
import time

A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"
MIN_FREE_MIB = 45000  # Need ~45 GiB free for v30a (d=128, batch 16).
POLL_INTERVAL = 60  # seconds

LAUNCH_CMD = (
    "cd {repo} && "
    "tmux has-session -t v30a_gpu1 2>/dev/null || "
    "tmux new-session -d -s v30a_gpu1 -n v30 'bash /tmp/launch_v30a.sh'"
)


def a800_ssh(cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    while True:
        out = a800_ssh(
            "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        )
        free_gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                free_gpus.append((int(parts[0]), int(parts[1])))

        candidates = [(g, f) for g, f in free_gpus if f >= MIN_FREE_MIB]
        if candidates:
            gpu, free = candidates[0]
            print(f"GPU {gpu} has {free} MiB free; launching v30a.")
            # launch_v30a.sh is expected to already exist on A800.
            a800_ssh(LAUNCH_CMD.format(repo=A800_REPO))
            print("v30a launched.")
            break
        else:
            print(
                f"No GPU with >= {MIN_FREE_MIB} MiB free; "
                f"sleeping {POLL_INTERVAL}s"
            )
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
