#!/usr/bin/env python
"""Poll GPU utilisation/processes and launch a command once it is idle.

Usage:
    python scripts/launch_when_gpu_free.py --gpu 0 --free-mib 1000 -- bash my_run.sh

The script waits until the specified GPU has at least ``free_mib`` MiB of free
memory, GPU utilisation below ``max-util``, and no other processes running on
it.  Designed to queue long training runs on a single local GPU without manual
babysitting.
"""

import argparse
import subprocess
import sys
import time
from typing import Optional, Tuple


def query_gpu(gpu: int) -> Tuple[int, int, int]:
    """Return (free_mib, util_pct, process_count) for ``gpu``.

    Falls back to (0, 100, 999) if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        line = out.strip().split("\n")[0].strip()
        free_str, util_str = line.split(",")
        free_mib = int(free_str.strip())
        util_pct = int(util_str.strip())
    except Exception:
        return 0, 100, 999

    try:
        proc_out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        proc_count = len([line for line in proc_out.strip().split("\n") if line.strip()])
    except Exception:
        proc_count = 0

    return free_mib, util_pct, proc_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a command once a GPU is idle.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--free-mib", type=int, default=1000, help="Required free memory in MiB")
    parser.add_argument("--max-util", type=int, default=5, help="Maximum GPU utilisation percent to consider idle")
    parser.add_argument("--poll-sec", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run once GPU is free")
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print(f"Waiting for GPU {args.gpu} to be idle (free >= {args.free_mib} MiB, util <= {args.max_util}%, no processes)...")
    while True:
        free_mib, util_pct, proc_count = query_gpu(args.gpu)
        print(f"GPU {args.gpu}: free={free_mib} MiB, util={util_pct}%, processes={proc_count}")
        if free_mib >= args.free_mib and util_pct <= args.max_util and proc_count == 0:
            break
        time.sleep(args.poll_sec)

    print(f"GPU {args.gpu} is idle; running: {' '.join(args.command)}")
    sys.exit(subprocess.run(args.command).returncode)


if __name__ == "__main__":
    main()
