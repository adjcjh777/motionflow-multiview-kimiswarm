#!/usr/bin/env python
"""Poll GPU memory utilisation and launch a command once it drops.

Usage:
    python scripts/launch_when_gpu_free.py --gpu 0 --free-mib 1000 -- bash my_run.sh

The script waits until the specified GPU has at least ``free_mib`` MiB of free
memory, then execs the supplied command.  Designed to queue long training runs
on a single local GPU without manual babysitting.
"""

import argparse
import subprocess
import sys
import time


def get_gpu_free_mib(gpu: int) -> int:
    """Return free memory in MiB for ``gpu`` using nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return int(out.strip().split("\n")[0].strip())
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a command once a GPU is free.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--free-mib", type=int, default=1000, help="Required free memory in MiB")
    parser.add_argument("--poll-sec", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run once GPU is free")
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print(f"Waiting for GPU {args.gpu} to have >= {args.free_mib} MiB free...")
    while True:
        free = get_gpu_free_mib(args.gpu)
        print(f"GPU {args.gpu} free memory: {free} MiB")
        if free >= args.free_mib:
            break
        time.sleep(args.poll_sec)

    print(f"GPU {args.gpu} is free; running: {' '.join(args.command)}")
    sys.exit(subprocess.run(args.command).returncode)


if __name__ == "__main__":
    main()
