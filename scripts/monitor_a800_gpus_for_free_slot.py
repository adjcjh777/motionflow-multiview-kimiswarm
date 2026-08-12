#!/usr/bin/env python3
"""Monitor A800-D GPUs 6/7 and notify when a slot becomes free.

This script runs locally (or anywhere with SSH access to a800-D). It polls GPU
memory/utilization on the A800 host and emits a notification when at least one
of the project GPUs (6 or 7) drops below the configured thresholds.

Supported notification channels:

* Local log file (always)
* Email via SMTP (optional)
* Slack incoming webhook (optional)

Environment
-----------
FREE_GPU_EMAIL_TO
    Comma-separated list of recipient email addresses.
FREE_GPU_EMAIL_SMTP_HOST
    SMTP server hostname (default: smtp.gmail.com).
FREE_GPU_EMAIL_SMTP_PORT
    SMTP server port (default: 587).
FREE_GPU_EMAIL_SMTP_USER
    SMTP login user. If unset, email notifications are skipped.
FREE_GPU_EMAIL_SMTP_PASS
    SMTP login password. If unset, email notifications are skipped.
FREE_GPU_SLACK_WEBHOOK
    Slack incoming-webhook URL. If unset, Slack notifications are skipped.

Usage
-----
    python scripts/monitor_a800_gpus_for_free_slot.py --interval 60 --memory-mb 5000

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_SSH_HOST = "a800-D"
DEFAULT_A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "monitor_a800_gpus_for_free_slot.log"
DEFAULT_POLL_INTERVAL = 60  # seconds
DEFAULT_COOLDOWN = 3600  # seconds between repeat notifications for the same GPU
DEFAULT_FREE_MEMORY_MB = 5000
DEFAULT_MAX_UTIL_PCT = 0

# Hard project GPU policy: only GPU 6 and GPU 7 are ever used.
TARGET_GPUS = (6, 7)


@dataclass
class GpuInfo:
    index: int
    util_pct: int
    memory_used_mb: int
    memory_total_mb: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


@dataclass
class NotificationState:
    last_alert_ts: float = 0.0
    was_free: bool = False


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def a800_ssh(cmd: str, ssh_host: str) -> str:
    """Run a command on a800-D via SSH and return stdout."""
    return subprocess.check_output(
        [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            ssh_host,
            cmd,
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )


def query_gpus(ssh_host: str) -> List[GpuInfo]:
    """Query nvidia-smi on a800-D and return GPU info for all GPUs."""
    out = a800_ssh(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits",
        ssh_host,
    )
    gpus: List[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    util_pct=int(float(parts[1].replace("%", ""))),
                    memory_used_mb=int(float(parts[2])),
                    memory_total_mb=int(float(parts[3])),
                )
            )
        except ValueError:
            continue
    return gpus


def find_free_gpus(
    gpus: Sequence[GpuInfo],
    memory_threshold_mb: int,
    max_util_pct: int,
) -> List[int]:
    """Return indices of GPUs that meet the free-slot criteria."""
    free: List[int] = []
    for gpu in gpus:
        if gpu.index not in TARGET_GPUS:
            continue
        if gpu.memory_used_mb < memory_threshold_mb and gpu.util_pct <= max_util_pct:
            free.append(gpu.index)
    return free


def build_message(free_gpus: List[int], all_gpus: List[GpuInfo], ssh_host: str) -> str:
    """Build a human-readable notification message."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"A800-D GPU slot free @ {now}",
        f"Free project GPUs: {', '.join(str(g) for g in free_gpus)}",
        "GPU status:",
    ]
    for gpu in sorted(all_gpus, key=lambda g: g.index):
        if gpu.index in TARGET_GPUS:
            lines.append(
                f"  GPU {gpu.index}: util={gpu.util_pct}%, "
                f"mem_used={gpu.memory_used_mb} MiB / {gpu.memory_total_mb} MiB"
            )
    lines.append(f"Host: {ssh_host}")
    return "\n".join(lines)


def send_email(subject: str, body: str, dry_run: bool = False) -> bool:
    """Send a notification email using SMTP credentials from the environment."""
    to_addrs = os.environ.get("FREE_GPU_EMAIL_TO", "").strip()
    smtp_host = os.environ.get("FREE_GPU_EMAIL_SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.environ.get("FREE_GPU_EMAIL_SMTP_PORT", "587"))
    smtp_user = os.environ.get("FREE_GPU_EMAIL_SMTP_USER", "").strip()
    smtp_pass = os.environ.get("FREE_GPU_EMAIL_SMTP_PASS", "").strip()

    if not to_addrs or not smtp_user or not smtp_pass:
        logging.debug("Email skipped: missing FREE_GPU_EMAIL_TO/SMTP_USER/SMTP_PASS.")
        return False

    if dry_run:
        logging.info("DRY-RUN: would send email to %s", to_addrs)
        return True

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addrs

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_addrs.split(","), msg.as_string())
        logging.info("Email notification sent to %s", to_addrs)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("Email notification failed: %s", exc)
        return False


def send_slack(body: str, dry_run: bool = False) -> bool:
    """Post a notification to a Slack incoming webhook."""
    webhook_url = os.environ.get("FREE_GPU_SLACK_WEBHOOK", "").strip()
    if not webhook_url:
        logging.debug("Slack skipped: FREE_GPU_SLACK_WEBHOOK not set.")
        return False

    if dry_run:
        logging.info("DRY-RUN: would post to Slack webhook")
        return True

    payload = json.dumps({"text": body}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "motionflow-gpu-monitor"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status in (200, 201)
    except Exception as exc:  # noqa: BLE001
        logging.error("Slack notification failed: %s", exc)
        return False


def notify(
    free_gpus: List[int],
    all_gpus: List[GpuInfo],
    ssh_host: str,
    dry_run: bool = False,
) -> None:
    """Emit notifications for a free slot via all configured channels."""
    subject = f"A800-D GPU slot free: {free_gpus}"
    message = build_message(free_gpus, all_gpus, ssh_host)

    logging.info("%s\n%s", subject, message)

    # Email subject line; body is the full message.
    send_email(subject, message, dry_run=dry_run)
    # Slack post is the full message.
    send_slack(message, dry_run=dry_run)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor A800-D GPUs 6/7 and notify when a slot becomes free."
    )
    parser.add_argument(
        "--ssh-host",
        default=DEFAULT_SSH_HOST,
        help="SSH host alias for A800-D (default: a800-D).",
    )
    parser.add_argument(
        "--a800-repo",
        default=DEFAULT_A800_REPO,
        help="A800-D training repo path (used only for context/logging).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=DEFAULT_FREE_MEMORY_MB,
        help=f"Memory threshold in MiB (default: {DEFAULT_FREE_MEMORY_MB}).",
    )
    parser.add_argument(
        "--max-util-pct",
        type=int,
        default=DEFAULT_MAX_UTIL_PCT,
        help=f"Maximum GPU utilization percent to be considered free (default: {DEFAULT_MAX_UTIL_PCT}).",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=DEFAULT_COOLDOWN,
        help=f"Seconds before re-notifying for the same GPU (default: {DEFAULT_COOLDOWN}).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to local log file.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single polling cycle and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions but do not send email/Slack notifications.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_path)

    logging.info(
        "Starting A800 GPU free-slot monitor (GPUs %s) every %ss; "
        "memory<%d MiB, util<=%d%%.",
        ", ".join(str(g) for g in TARGET_GPUS),
        args.interval,
        args.memory_mb,
        args.max_util_pct,
    )

    # Per-GPU state for cooldown / transition detection.
    state: Dict[int, NotificationState] = {g: NotificationState() for g in TARGET_GPUS}

    try:
        while True:
            try:
                gpus = query_gpus(args.ssh_host)
            except subprocess.CalledProcessError as exc:
                logging.error("Failed to query A800 GPUs: %s", exc)
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            free_gpus = find_free_gpus(gpus, args.memory_mb, args.max_util_pct)
            now = time.time()

            newly_free: List[int] = []
            for gpu_idx in free_gpus:
                st = state[gpu_idx]
                if not st.was_free or (now - st.last_alert_ts) >= args.cooldown:
                    newly_free.append(gpu_idx)
                    st.last_alert_ts = now
                    st.was_free = True

            for gpu_idx in TARGET_GPUS:
                if gpu_idx not in free_gpus:
                    state[gpu_idx].was_free = False

            if newly_free:
                notify(newly_free, gpus, args.ssh_host, dry_run=args.dry_run)
            else:
                logging.debug(
                    "No newly free slot. Free GPUs: %s", free_gpus if free_gpus else "none"
                )

            if args.once:
                logging.info("Single-shot complete.")
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
