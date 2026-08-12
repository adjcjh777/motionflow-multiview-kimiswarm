#!/usr/bin/env python3
"""Wait for all detected-2D MPI-INF-3DHP .npz files, then run conf-weighted DLT.

This companion script polls the output directory of
``scripts/generate_mpi_detected_2d.py`` (or the RTMPose AVI variant) until every
expected .npz has been written, then invokes ``scripts/run_mpi_dlt_baseline.py``
once and writes a JSON summary.

Usage
-----
    nohup python3 scripts/wait_and_run_mpi_dlt_baseline.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --detected_dir data/webbridge/mpi_inf_3dhp_detected_2d \
        --poll_interval 60 \
        --device cpu \
        --output_json outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json \
        > outputs/mpi_rtmpose_detected_2d/wait_and_run_dlt.log 2>&1 &
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def _expected_basenames(input_dir: Path) -> list[str]:
    """Return basenames of canonical source .npz files that will be regenerated."""
    # Mirrors the filter in scripts/generate_mpi_detected_2d.py.
    omit_re = re.compile(r"_smoke|_v4_|test_set")
    files = sorted(p for p in input_dir.glob("*.npz") if not omit_re.search(p.name))
    if not files:
        raise FileNotFoundError(f"No canonical .npz files found in {input_dir}")
    return [p.name for p in files]


def _missing_files(detected_dir: Path, expected: list[str]) -> list[str]:
    present = {p.name for p in detected_dir.glob("*.npz") if p.stat().st_size > 0}
    return [name for name in expected if name not in present]


def wait_for_files(
    input_dir: Path,
    detected_dir: Path,
    poll_interval: int,
    stable_rounds: int = 2,
) -> None:
    expected = _expected_basenames(input_dir)
    print(f"Waiting for {len(expected)} detected-2D files in {detected_dir}")
    print(f"  expected: {', '.join(expected)}")

    stable_count = 0
    last_missing: list[str] = expected[:]

    while True:
        missing = _missing_files(detected_dir, expected)
        present_count = len(expected) - len(missing)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{present_count}/{len(expected)} files ready"
        )
        if not missing:
            if stable_count >= stable_rounds:
                print("All files present and stable; launching DLT baseline.")
                break
            stable_count += 1
            print(f"  all present, waiting {stable_count}/{stable_rounds} stable checks...")
        else:
            if missing != last_missing:
                print(f"  still missing: {', '.join(missing)}")
            stable_count = 0
        last_missing = missing
        time.sleep(poll_interval)


def run_dlt(detected_dir: Path, device: str, output_json: Path) -> int:
    """Invoke the MPI DLT baseline script once all files are ready."""
    repo_root = Path(__file__).resolve().parents[1]
    glob_pattern = str(detected_dir / "*.npz")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_mpi_dlt_baseline.py"),
        "--glob", glob_pattern,
        "--device", device,
        "--output", str(output_json),
    ]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(repo_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for MPI detected-2D .npz files, then run DLT.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp"),
                        help="Directory containing canonical source .npz files.")
    parser.add_argument("--detected_dir", type=Path, default=Path("data/webbridge/mpi_inf_3dhp_detected_2d"),
                        help="Directory where detected-2D .npz files are being written.")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between polling checks.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device for DLT (cpu or cuda).")
    parser.add_argument("--output_json", type=Path,
                        default=Path("outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json"),
                        help="Where to write the DLT results JSON.")
    args = parser.parse_args()

    args.detected_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    wait_for_files(args.input_dir, args.detected_dir, args.poll_interval)
    return run_dlt(args.detected_dir, args.device, args.output_json)


if __name__ == "__main__":
    raise SystemExit(main())
