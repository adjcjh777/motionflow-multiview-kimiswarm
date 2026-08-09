"""Parse local and A800 run logs to build a val_MPJPE leaderboard."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def _extract_best_val_mpjpe(text: str) -> str | None:
    """Return the most recent 'Best val MPJPE' value from log text."""
    # Lines like: Best val MPJPE: 83.69mm -> outputs/...
    matches = re.findall(r"Best val MPJPE:\s*([0-9.]+)mm", text)
    if matches:
        return matches[-1]
    return None


def _extract_all_val_mpjpe(text: str) -> list[float]:
    """Return all 'val_MPJPE' values from log text."""
    return [float(v) for v in re.findall(r"val_MPJPE=([0-9.]+)mm", text)]


def _extract_latest_val_mpjpe(text: str) -> str | None:
    """Return the latest 'val_MPJPE' value from log text."""
    values = _extract_all_val_mpjpe(text)
    if values:
        return str(values[-1])
    return None


def _extract_best_val_mpjpe(text: str) -> str | None:
    """Return the best (lowest) 'val_MPJPE' value from log text."""
    values = _extract_all_val_mpjpe(text)
    if values:
        return str(min(values))
    return None


def _parse_local_outputs(outputs_dir: Path) -> list[dict]:
    results: list[dict] = []
    for log_path in outputs_dir.glob("*.log"):
        text = log_path.read_text(errors="ignore")
        best = _extract_best_val_mpjpe(text)
        latest = _extract_latest_val_mpjpe(text)
        if best or latest:
            results.append(
                {
                    "run": log_path.stem,
                    "best_val_mpjpe_mm": best or "",
                    "latest_val_mpjpe_mm": latest or "",
                    "location": "local",
                }
            )
    return results


def _parse_a800_outputs(ssh_host: str, a800_repo: str) -> list[dict]:
    cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "BatchMode=yes",
        ssh_host,
        f"find {a800_repo}/outputs -maxdepth 1 -name '*.log' -print0 | xargs -0 grep -H 'val_MPJPE=' 2>/dev/null || true",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, errors="ignore")
    except subprocess.CalledProcessError:
        return []

    results_by_run: dict[str, dict] = {}
    for line in out.splitlines():
        # line format: path: ... val_MPJPE=XX.XXmm
        if "val_MPJPE=" not in line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        log_path = Path(parts[0].strip())
        rest = ":".join(parts[1:])
        run_name = log_path.stem
        values = _extract_all_val_mpjpe(rest)
        if not values:
            continue
        if run_name not in results_by_run:
            results_by_run[run_name] = {
                "run": run_name,
                "best_val_mpjpe_mm": str(min(values)),
                "latest_val_mpjpe_mm": str(values[-1]),
                "location": "a800",
            }
        else:
            current_best = float(results_by_run[run_name]["best_val_mpjpe_mm"] or "inf")
            results_by_run[run_name]["best_val_mpjpe_mm"] = str(min(current_best, *values))
            results_by_run[run_name]["latest_val_mpjpe_mm"] = str(values[-1])
    return list(results_by_run.values())


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "No completed runs with val_MPJPE found."
    header = ["Run", "Best val_MPJPE (mm)", "Latest val_MPJPE (mm)", "Location"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [row["run"], row["best_val_mpjpe_mm"], row["latest_val_mpjpe_mm"], row["location"]]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse run results and print a leaderboard.")
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"), help="Local outputs directory")
    parser.add_argument("--ssh_host", default="a800-D", help="A800 SSH host")
    parser.add_argument("--a800_repo", default="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20", help="A800 repo path")
    args = parser.parse_args()

    rows = _parse_local_outputs(args.outputs_dir)
    rows.extend(_parse_a800_outputs(args.ssh_host, args.a800_repo))

    # Sort by best val MPJPE (numeric, missing values last)
    def _key(row: dict) -> float:
        try:
            return float(row["best_val_mpjpe_mm"]) if row["best_val_mpjpe_mm"] else float("inf")
        except ValueError:
            return float("inf")

    rows.sort(key=_key)

    print(_markdown_table(rows))


if __name__ == "__main__":
    main()
