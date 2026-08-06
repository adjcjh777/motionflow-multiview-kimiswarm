"""Write a calibration-robustness markdown table from the PP eval JSON.

Usage
-----
    python experiments/write_pp_robustness_table.py \
        --input outputs/crossview_pp_full_ppw005_20ep_eval.json \
        --output docs/tables/icra2027/robustness.md
"""

import argparse
import json
from pathlib import Path


def make_table(src: dict) -> str:
    robustness = src.get("robustness", {})
    lines = [
        "| Condition | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["clean", "rot_0.5_deg", "rot_1.0_deg", "trans_5mm", "trans_10mm",
                 "focal_1pct", "focal_2pct", "cxcy_3px", "cxcy_5px"]:
        if name not in robustness:
            continue
        r = robustness[name]
        lines.append(
            f"| {name} | {r['mpjpe']:.2f} | {r['pa_mpjpe']:.2f} | "
            f"{r['pck@50mm']:.2f} | {r['pck@100mm']:.2f} | {r['pck@150mm']:.2f} | {r['pck_auc']:.3f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    src = json.loads(Path(args.input).read_text())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(make_table(src))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
