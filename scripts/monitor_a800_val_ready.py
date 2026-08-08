#!/usr/bin/env python3
"""A800 training monitor: poll training logs and post val_MPJPE to GitHub.

This script runs locally (or anywhere with SSH access to a800-D) and polls the
``outputs/omniview_fusion_v*.log`` files on the A800-D training repo.  When a
new validation MPJPE line appears, it posts a short summary comment to the
configured GitHub issue so the self-evolution loop stays documented without
manual checks.

Usage
-----
    python scripts/monitor_a800_val_ready.py --issue 88 --poll-interval 120

Environment
-----------
GITHUB_TOKEN
    Falls back to extracting the token from ``git remote get-url origin``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import urllib.request


DEFAULT_A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
DEFAULT_SSH_HOST = "a800-D"
DEFAULT_ISSUE = 88
DEFAULT_POLL_INTERVAL = 120  # seconds
MAX_LOG_AGE_MINUTES = 120  # only monitor logs modified in the last 2 hours
DEFAULT_STATE_PATH = ".monitor_a800_state.json"
DEFAULT_INCLUDE_PATTERN = ".*"


@dataclass
class RunState:
    last_val_step: int = -1
    best_val: Optional[float] = None


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def get_github_token() -> Optional[str]:
    """Return a GitHub token from env or from the local git remote URL."""
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


def a800_ssh(cmd: str, ssh_host: str = DEFAULT_SSH_HOST) -> str:
    """Run a command on a800-D and return stdout."""
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def load_state(state_path: Path) -> Dict[str, RunState]:
    """Load per-run monitoring state from disk."""
    if not state_path.exists():
        return {}
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        return {k: RunState(**v) for k, v in raw.items()}
    except Exception as exc:
        _log(f"Warning: could not load state ({exc}); starting fresh.")
        return {}


def save_state(state: Dict[str, RunState], state_path: Path) -> None:
    """Persist per-run monitoring state to disk."""
    raw = {k: {"last_val_step": v.last_val_step, "best_val": v.best_val} for k, v in state.items()}
    state_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def parse_val_lines(text: str) -> List[Tuple[int, float]]:
    """Return (step, mpjpe_mm) tuples for each val_MPJPE line in *text*."""
    out: List[Tuple[int, float]] = []
    for line in text.splitlines():
        # Match patterns like "Epoch 1: val_MPJPE=23.45mm" or "val_MPJPE=23.45mm"
        step_match = re.search(r"(?:train step |Epoch |step )?(\d+).*val_MPJPE=([\d.]+)mm", line, re.I)
        if step_match:
            step = int(step_match.group(1))
            mpjpe = float(step_match.group(2))
            out.append((step, mpjpe))
    return out


def discover_logs(repo_dir: str, ssh_host: str, max_age_minutes: int = MAX_LOG_AGE_MINUTES, include_pattern: str = DEFAULT_INCLUDE_PATTERN) -> List[str]:
    """Return list of recently modified log file names on A800-D."""
    out = a800_ssh(
        f"find {repo_dir}/outputs -maxdepth 1 -name '*.log' -mmin -{max_age_minutes} 2>/dev/null",
        ssh_host,
    )
    compiled = re.compile(include_pattern)
    return [Path(p).name for p in out.strip().split("\n") if p.strip() and compiled.match(Path(p).name)]


def fetch_log(repo_dir: str, log_name: str, ssh_host: str) -> str:
    return a800_ssh(f"cat {repo_dir}/outputs/{log_name}", ssh_host)


def fetch_tail(repo_dir: str, log_name: str, n: int, ssh_host: str) -> str:
    return a800_ssh(f"tail -n {n} {repo_dir}/outputs/{log_name}", ssh_host)


def post_issue_comment(token: str, issue: int, body: str, repo: str = "adjcjh777/motionflow-multiview-kimiswarm") -> bool:
    """Post a comment to the configured GitHub issue."""
    url = f"https://api.github.com/repos/{repo}/issues/{issue}/comments"
    req = urllib.request.Request(
        url,
        data=json.dumps({"body": body}).encode("utf-8"),
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "User-Agent": "motionflow-monitor",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status == 201
    except Exception as exc:
        _log(f"GitHub comment failed: {exc}")
        return False


def build_summary(name: str, vals: List[Tuple[int, float]]) -> str:
    best_step, best_val = min(vals, key=lambda x: x[1])
    last_step, last_val = vals[-1]
    return f"- **{name}**: best {best_val:.2f} mm at step {best_step}, latest {last_val:.2f} mm at step {last_step}"


def poll_once(
    repo_dir: str,
    ssh_host: str,
    issue: int,
    token: str,
    state: Dict[str, RunState],
    dry_run: bool,
    include_pattern: str = DEFAULT_INCLUDE_PATTERN,
) -> Dict[str, RunState]:
    """Run one polling cycle and return updated state."""
    log_names = discover_logs(repo_dir, ssh_host, include_pattern=include_pattern)
    new_state: Dict[str, RunState] = {}
    notifications: List[str] = []

    for name in log_names:
        try:
            text = fetch_tail(repo_dir, name, 200, ssh_host)
        except subprocess.CalledProcessError as exc:
            _log(f"Could not fetch {name}: {exc}")
            continue

        vals = parse_val_lines(text)
        if not vals:
            continue

        prev = state.get(name, RunState())
        new_vals = [(s, m) for s, m in vals if s > prev.last_val_step]
        if new_vals:
            summary = build_summary(name, new_vals)
            notifications.append(summary)
            _log(f"New val_MPJPE for {name}: {summary}")

        best = min(vals, key=lambda x: x[1])
        new_state[name] = RunState(last_val_step=vals[-1][0], best_val=best[1])

    if notifications and not dry_run:
        body = "## Auto-detected validation MPJPE updates\n\n" + "\n".join(notifications)
        post_issue_comment(token, issue, body)
    elif notifications and dry_run:
        _log("DRY-RUN would post:\n" + "\n".join(notifications))

    return new_state


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor A800 training and post val_MPJPE to GitHub.")
    parser.add_argument("--a800-repo", default=DEFAULT_A800_REPO, help="A800-D training repo path")
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST, help="SSH host alias for A800-D")
    parser.add_argument("--issue", type=int, default=DEFAULT_ISSUE, help="GitHub issue number to post to")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Seconds between polls")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, help="Path to the per-run monitoring state file")
    parser.add_argument("--include-pattern", default=DEFAULT_INCLUDE_PATTERN, help="Regex pattern to filter log file names")
    parser.add_argument("--once", action="store_true", help="Run a single polling cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Log actions but do not post to GitHub")
    args = parser.parse_args(argv)

    token = get_github_token()
    if not token:
        _log("No GitHub token found. Set GITHUB_TOKEN or include it in the git remote URL.")
        return 1

    state_path = Path(args.state)
    state = load_state(state_path)
    _log(f"Monitoring A800-D logs every {args.poll_interval}s (issue #{args.issue}, state={state_path}).")

    try:
        while True:
            state = poll_once(args.a800_repo, args.ssh_host, args.issue, token, state, args.dry_run, include_pattern=args.include_pattern)
            save_state(state, state_path)
            if args.once:
                _log("Single-shot complete.")
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        _log("Interrupted.")
    finally:
        save_state(state, state_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
