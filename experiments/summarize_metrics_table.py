"""Aggregate all metrics JSON files into a single Markdown table."""

import json
import glob
from pathlib import Path


def main():
    files = sorted(glob.glob("outputs/metrics_*.json"))
    rows = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        name = Path(f).stem.replace("metrics_", "").replace("_", " ")
        rows.append({
            "name": name,
            "mpjpe": data.get("mpjpe", float("nan")),
            "pa_mpjpe": data.get("pa_mpjpe", float("nan")),
            "pck_50": data.get("pck@50mm", float("nan")),
            "pck_100": data.get("pck@100mm", float("nan")),
            "pck_150": data.get("pck@150mm", float("nan")),
            "auc": data.get("pck_auc", float("nan")),
        })

    print("| Model | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['name']} | {row['mpjpe']:.2f} | {row['pa_mpjpe']:.2f} | "
            f"{row['pck_50']:.4f} | {row['pck_100']:.4f} | {row['pck_150']:.4f} | {row['auc']:.4f} |"
        )


if __name__ == "__main__":
    main()
