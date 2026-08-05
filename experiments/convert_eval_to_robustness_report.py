"""Convert the robustness section of a PP eval JSON to the report schema used by paper_figures/tables.

Usage
-----
    python experiments/convert_eval_to_robustness_report.py \
        --input outputs/crossview_pp_full_ppw005_20ep_eval.json \
        --output outputs/robustness_pp_best_mpi.json
"""

import argparse
import json
from pathlib import Path


def convert(src: dict) -> dict:
    robustness = src.get("robustness", {})
    noise_entries = []
    occlusion_entries = []
    outliers_entries = []

    # Calibration-robustness entries from the PP eval
    for name, report in robustness.items():
        if name == "clean":
            continue
        mpjpe = report.get("mpjpe")
        if mpjpe is None:
            continue
        # Map to legacy labels for the table generator
        if name.startswith("rot_"):
            noise_entries.append({"noise_std_px": name, "mpjpe_mm": mpjpe})
        elif name.startswith("trans_"):
            occlusion_entries.append({"occlusion_rate": name, "mpjpe_mm": mpjpe})
        elif name.startswith("focal_") or name.startswith("cxcy_"):
            outliers_entries.append({"outlier_rate": name, "mpjpe_mm": mpjpe})

    # Keep a generic structure so the table can render cleanly
    report = {
        "noise": noise_entries,
        "occlusion": occlusion_entries,
        "outliers": outliers_entries,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    src = json.loads(Path(args.input).read_text())
    report = convert(src)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
