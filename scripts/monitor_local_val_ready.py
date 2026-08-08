#!/usr/bin/env python
"""Monitor local training logs for val_MPJPE and emit concise status updates.

Usage:
    python scripts/monitor_local_val_ready.py --log outputs/omniview_fusion_v26_udp_full_local_4090.log
"""
import argparse
import re
import time
from pathlib import Path


def parse_log(path: str) -> dict:
    """Return best and latest val_MPJPE (mm) and corresponding epoch."""
    best_val = float("inf")
    best_epoch = None
    latest_val = None
    latest_epoch = None

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # Match lines like: Epoch 3: train_loss=..., val_loss=..., val_MPJPE=60.37mm
    for m in re.finditer(
        r"Epoch\s+(\d+):\s+train_loss=[\d\.]+,\s+val_loss=[\d\.]+,\s+val_MPJPE=([\d\.]+)mm",
        text,
    ):
        epoch = int(m.group(1))
        val = float(m.group(2))
        latest_val = val
        latest_epoch = epoch
        if val < best_val:
            best_val = val
            best_epoch = epoch

    return {
        "best_val": best_val if best_val != float("inf") else None,
        "best_epoch": best_epoch,
        "latest_val": latest_val,
        "latest_epoch": latest_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor local training log for val_MPJPE.")
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--poll-sec", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists() and args.once:
        print(f"Log not found: {log_path}")
        return

    if args.once:
        info = parse_log(str(log_path))
        print(
            f"[{log_path.name}] best={info['best_val']}mm @ epoch {info['best_epoch']} | "
            f"latest={info['latest_val']}mm @ epoch {info['latest_epoch']}"
        )
        return

    last_size = 0
    print(f"Monitoring {log_path} every {args.poll_sec}s...")
    while True:
        try:
            current_size = log_path.stat().st_size
            if current_size != last_size:
                last_size = current_size
                info = parse_log(str(log_path))
                if info["latest_val"] is not None:
                    print(
                        f"[{log_path.name}] best={info['best_val']}mm @ epoch {info['best_epoch']} | "
                        f"latest={info['latest_val']}mm @ epoch {info['latest_epoch']}"
                    )
        except FileNotFoundError:
            pass
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
