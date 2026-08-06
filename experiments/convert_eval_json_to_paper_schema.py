"""Convert a nested evaluation JSON to the flat schema expected by the paper figure/table pipeline.

Usage
-----
    python experiments/convert_eval_json_to_paper_schema.py \
        --input outputs/crossview_pp_full_ppw005_20ep_eval.json \
        --output outputs/eval_jsons/pp_best_mpi.json \
        --name "PP (clean)"
"""

import argparse
import json
from pathlib import Path


def convert(src: dict, name: str) -> dict:
    clean = src.get("clean", {})
    # If the top-level already has mpjpe, use it; otherwise fall back to clean.
    mpjpe = clean.get("mpjpe", src.get("mpjpe"))
    pa_mpjpe = clean.get("pa_mpjpe", src.get("pa_mpjpe"))
    pck50 = clean.get("pck@50mm", src.get("pck@50mm"))
    pck100 = clean.get("pck@100mm", src.get("pck@100mm"))
    pck150 = clean.get("pck@150mm", src.get("pck@150mm"))
    auc = clean.get("pck_auc", src.get("pck_auc"))

    out = {
        "name": name,
        "mpjpe_mm": mpjpe,
        "pa_mpjpe_mm": pa_mpjpe,
        "pck_50": pck50,
        "pck_100": pck100,
        "pck_150": pck150,
        "auc": auc,
        # Per-joint/PCK-curve arrays are only available when the eval script is
        # run with --save_full. They are left as None here so that the figure
        # pipeline skips the optional per-joint/PCK plots.
        "per_joint_mpjpe": None,
        "per_joint_pa_mpjpe": None,
        "pck_thresholds": None,
        "pck_values": None,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert eval JSON to paper schema")
    parser.add_argument("--input", type=str, required=True, help="Input nested eval JSON")
    parser.add_argument("--output", type=str, required=True, help="Output flat JSON")
    parser.add_argument("--name", type=str, default=None, help="Optional model name")
    args = parser.parse_args()

    src = json.loads(Path(args.input).read_text())
    name = args.name or Path(args.input).stem
    out = convert(src, name)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
