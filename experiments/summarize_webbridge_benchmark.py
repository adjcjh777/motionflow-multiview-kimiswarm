"""Summarize a WebBridge benchmark JSON into a Markdown table.

Usage:
    python experiments/summarize_webbridge_benchmark.py \
        --json outputs/webbridge_benchmark_crossview_residual_smoke_v2.json \
        --out docs/results_webbridge_crossview_residual.md
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Benchmark JSON path")
    parser.add_argument("--out", type=str, default=None, help="Optional Markdown output path")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    rows = data.get("results", [])
    if not rows:
        print("No results in JSON.")
        return

    header = "| Dataset | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    lines = ["# WebBridge Benchmark Summary\n", f"Manifest: `{data.get('manifest')}`\n", header, separator]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | "
            f"{r['mpjpe_mm']:.2f} | "
            f"{r['pa_mpjpe_mm']:.2f} | "
            f"{r['pck_50']:.4f} | "
            f"{r['pck_100']:.4f} | "
            f"{r['pck_150']:.4f} | "
            f"{r['pck_auc']:.4f} |"
        )
    md = "\n".join(lines)
    print(md)

    if args.out:
        Path(args.out).write_text(md + "\n")
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
