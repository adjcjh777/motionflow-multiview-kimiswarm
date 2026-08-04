"""Regenerate all paper figures and tables from evaluation JSONs.

Usage
-----
    conda run -n mf python experiments/generate_paper_figures_and_tables.py \
        --eval_dir outputs/eval_jsons \
        --robustness outputs/robustness_residual_final5/robustness_report.json \
        --out_dir docs/figures/icra2027 \
        --table_dir docs/tables/icra2027

Dependencies
------------
    numpy, matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.lib import paper_figures as pf
from experiments.lib import paper_tables as pt


FIG_DIR = Path("docs/figures/icra2027")
TABLE_DIR = Path("docs/tables/icra2027")


def load_evals(eval_dir: Path) -> dict:
    """Load every *.json in eval_dir into a dict keyed by filename stem."""
    results = {}
    for p in sorted(eval_dir.glob("*.json")):
        results[p.stem] = json.loads(p.read_text())
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--eval_dir", type=str, default="outputs/eval_jsons",
                        help="Directory containing evaluation JSON files")
    parser.add_argument("--robustness", type=str, default=None,
                        help="Path to robustness_report.json")
    parser.add_argument("--out_dir", type=str, default=str(FIG_DIR))
    parser.add_argument("--table_dir", type=str, default=str(TABLE_DIR))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    table_dir = Path(args.table_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if not eval_dir.exists():
        print(f"ERROR: eval_dir {eval_dir} does not exist.")
        sys.exit(1)

    results = load_evals(eval_dir)
    if not results:
        print(f"WARNING: no JSON files found in {eval_dir}")

    # 1. Main MPJPE bar chart
    if results:
        pf.draw_main_mpjpe_bar(results, out_dir / "main_mpjpe_bar.png")

    # 2. Per-joint MPJPE for the best model
    for name, r in results.items():
        per_joint = r.get("per_joint_mpjpe")
        if per_joint is not None:
            pf.draw_per_joint_mpjpe(
                np.asarray(per_joint) * 1000.0,  # assume stored in meters
                out_dir / f"per_joint_mpjpe_{name}.png",
                joint_names=pf.SKELETON_JOINT_NAMES_17,
            )
            break  # only the first model with per-joint data

    # 3. PCK curve for the best model
    for name, r in results.items():
        thresholds = r.get("pck_thresholds")
        pck_values = r.get("pck_values")
        if thresholds is not None and pck_values is not None:
            pf.draw_pck_curve(
                np.asarray(thresholds) * 1000.0,
                np.asarray(pck_values),
                out_dir / f"pck_curve_{name}.png",
                auc=r.get("auc"),
            )
            break

    # 4. Robustness grid
    if args.robustness and Path(args.robustness).exists():
        robustness_report = json.loads(Path(args.robustness).read_text())
        pf.draw_robustness_grid(robustness_report, out_dir / "robustness_grid.png")

    # 5. Tables
    pt.write_tables(results, table_dir)

    print("Generated files:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  figure: {f}")
    for f in sorted(table_dir.glob("*")):
        print(f"  table:  {f}")


if __name__ == "__main__":
    main()
