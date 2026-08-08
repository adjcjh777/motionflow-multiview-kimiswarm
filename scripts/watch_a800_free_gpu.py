#!/usr/bin/env python3
"""Watch A800-D GPUs and auto-launch v24 when one becomes free.

Runs locally, polls `nvidia-smi` on a800-D, and launches the prepared v24 small
script in a tmux session on the first free GPU.  Posts a comment to the
tracking issue so the action is recorded.

Usage:
    python scripts/watch_a800_free_gpu.py --issue 91 --poll-interval 60
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_SSH_HOST = "a800-D"
DEFAULT_A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
DEFAULT_ISSUE = 91
DEFAULT_POLL_INTERVAL = 60  # seconds
DEFAULT_FREE_MEMORY_MB = 5000


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def a800_ssh(cmd: str, ssh_host: str = DEFAULT_SSH_HOST) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def get_gpu_memory_used_mb(ssh_host: str) -> List[Tuple[int, int]]:
    """Return list of (index, memory_used_mb) for each GPU on a800-D."""
    out = a800_ssh(
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits", ssh_host
    )
    gpus: List[Tuple[int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append((int(parts[0]), int(parts[1])))
    return gpus


def is_training_on_gpu(gpu_index: int, ssh_host: str) -> bool:
    """Heuristic: check if a known training process is using this GPU."""
    try:
        out = a800_ssh(
            f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader --id={gpu_index}",
            ssh_host,
        )
    except subprocess.CalledProcessError:
        return False
    return "train_omniview_fusion" in out or "omniview_fusion" in out


def find_free_gpu(gpus: List[Tuple[int, int]], ssh_host: str) -> Optional[int]:
    """Return the first GPU that looks free and has no training process."""
    for idx, mem_mb in gpus:
        if mem_mb < DEFAULT_FREE_MEMORY_MB and not is_training_on_gpu(idx, ssh_host):
            return idx
    return None


def launch_v24(gpu: int, a800_repo: str, ssh_host: str) -> None:
    cmd = (
        f"cd {a800_repo} && "
        f"bash scripts/launch_v24_a800_tmux.sh {gpu}"
    )
    a800_ssh(cmd, ssh_host)


def get_github_token() -> Optional[str]:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        match = re.search(r"gho_[A-Za-z0-9]+", url)
        if match:
            return match.group(0)
    except Exception:
        pass
    return None


def post_issue_comment(token: str, issue: int, body: str) -> bool:
    url = f"https://api.github.com/repos/adjcjh777/motionflow-multiview-kimiswarm/issues/{issue}/comments"
    req = urllib.request.Request(
        url,
        data=json.dumps({"body": body}).encode("utf-8"),
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "User-Agent": "motionflow-watch",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status == 201
    except Exception as exc:
        _log(f"GitHub comment failed: {exc}")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Watch A800-D and auto-launch v24 on a free GPU.")
    parser.add_argument("--issue", type=int, default=DEFAULT_ISSUE)
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--a800-repo", default=DEFAULT_A800_REPO)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    token = get_github_token()
    if not token:
        _log("No GitHub token found. Set GITHUB_TOKEN or include it in git remote URL.")
        return 1

    _log(f"Watching A800-D GPUs every {args.poll_interval}s; will launch v24 on first free GPU.")

    while True:
        try:
            gpus = get_gpu_memory_used_mb(args.ssh_host)
        except subprocess.CalledProcessError as exc:
            _log(f"Could not query GPUs: {exc}")
            if args.once:
                break
            time.sleep(args.poll_interval)
            continue

        _log(f"GPU memory (MB): {gpus}")
        free_gpu = find_free_gpu(gpus, args.ssh_host)

        if free_gpu is not None:
            _log(f"GPU {free_gpu} is free; launching v24.")
            try:
                launch_v24(free_gpu, args.a800_repo, args.ssh_host)
                body = (
                    f"Auto-launched v24 (fixed BA + KAP) on A800-D GPU {free_gpu} "
                    f"via `scripts/watch_a800_free_gpu.py` at {time.strftime('%Y-%m-%dT%H:%M:%S')} UTC."
                )
                post_issue_comment(token, args.issue, body)
                _log("Launched v24.")
            except Exception as exc:
                _log(f"Launch failed: {exc}")
            if args.once:
                break
        else:
            _log("No free GPU.")

        if args.once:
            break
        time.sleep(args.poll_interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
