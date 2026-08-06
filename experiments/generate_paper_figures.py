"""Generate paper figures and tables from existing evaluation JSONs.

This script is a lightweight, CPU-only orchestrator.  It reads the evaluation
JSON files already produced under ``outputs/`` and writes camera-ready
figures and tables under ``docs/figures/icra2027/`` and ``docs/tables/icra2027/``.

Usage
-----
    .venv/Scripts/python experiments/generate_paper_figures.py

Optional arguments
------------------
    --eval_dir outputs
    --robustness outputs/robustness_residual_final5/robustness_report.json
    --out_dir docs/figures/icra2027
    --table_dir docs/tables/icra2027
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.lib import paper_figures as pf
from experiments.lib import paper_tables as pt


OUTPUTS_DIR = Path("outputs")
FIG_DIR = Path("docs/figures/icra2027")
TABLE_DIR = Path("docs/tables/icra2027")
ROBUSTNESS_DEFAULT = OUTPUTS_DIR / "robustness_residual_final5" / "robustness_report.json"

# Curated list of result files to show in the main comparison.
# Filename stem is used as the model label.
DEFAULT_RESULT_FILES = [
    "metrics_residual_mpiinf3dhp",
    "eval_residual_final5",
    "eval_crossview_residual_d64_h128",
    "eval_ray_attention_temporal_residual_mpiinf3dhp_eval",
    "metrics_campe_mpiinf3dhp",
    "metrics_adaptive_mpiinf3dhp",
    "eval_residual_h36m_h128",
    "metrics_campegraph_h36m_s5a02",
]


def _find_json(path_or_stem: str, eval_dir: Path) -> Path:
    """Resolve a bare stem or full/relative path to an JSON file."""
    p = Path(path_or_stem)
    if p.exists():
        return p
    candidate = eval_dir / f"{path_or_stem}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot find result file for stem: {path_or_stem}")


def _normalize_eval(raw: dict) -> dict:
    """Normalize heterogeneous eval JSONs to the schema expected by the libs."""
    normalized = {}

    def _get(*keys):
        for k in keys:
            if k in raw:
                return raw[k]
        return None

    # Best MPI result (10.46 mm) lives in metrics_residual_mpiinf3dhp.json
    mpjpe = _get("mpjpe_mm", "mpjpe")
    pa_mpjpe = _get("pa_mpjpe_mm", "pa_mpjpe")

    if mpjpe is None:
        raise ValueError("JSON missing mpjpe_mm/mpjpe field")

    normalized["mpjpe_mm"] = float(mpjpe)
    normalized["pa_mpjpe_mm"] = float(pa_mpjpe) if pa_mpjpe is not None else 0.0
    normalized["pck_50"] = float(
        _get("pck_50mm", "pck@50mm") or 0.0
    )
    normalized["pck_100"] = float(
        _get("pck_100mm", "pck@100mm") or 0.0
    )
    normalized["pck_150"] = float(
        _get("pck_150mm", "pck@150mm") or 0.0
    )
    auc = _get("pck_auc_150mm", "pck_auc")
    normalized["auc"] = float(auc) if auc is not None else 0.0

    per_joint = _get("per_joint_mpjpe_mm", "per_joint_mpjpe")
    if per_joint is not None:
        normalized["per_joint_mpjpe"] = np.asarray(per_joint, dtype=float)
    else:
        normalized["per_joint_mpjpe"] = None

    return normalized


def load_evals(eval_dir: Path, stems: list) -> dict:
    """Load and normalize the selected evaluation JSONs."""
    results = {}
    for stem in stems:
        try:
            path = _find_json(stem, eval_dir)
        except FileNotFoundError:
            print(f"[SKIP] {stem}.json not found in {eval_dir}")
            continue
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            print(f"[SKIP] {path.name} is a list, not a single eval dict")
            continue
        try:
            results[stem] = _normalize_eval(raw)
        except ValueError as exc:
            print(f"[SKIP] {path.name}: {exc}")
    return results


def make_robustness_markdown(robustness_report: dict) -> str:
    """Build a Markdown robustness table from the standardized report."""
    lines = [
        "| Perturbation | Level | MPJPE (mm) | PA-MPJPE (mm) |",
        "|---|---|---:|---:|",
    ]
    for entry in robustness_report.get("noise", []):
        lines.append(
            f"| Gaussian noise | {entry['noise_std_px']} px | {entry['mpjpe_mm']:.2f} | "
            f"{entry['pa_mpjpe_mm']:.2f} |"
        )
    for entry in robustness_report.get("occlusion", []):
        lines.append(
            f"| Joint occlusion | {entry['occlusion_rate']*100:.0f}% | {entry['mpjpe_mm']:.2f} | "
            f"{entry['pa_mpjpe_mm']:.2f} |"
        )
    for entry in robustness_report.get("outliers", []):
        lines.append(
            f"| 2D outliers | {entry['outlier_rate']*100:.0f}% | {entry['mpjpe_mm']:.2f} | "
            f"{entry['pa_mpjpe_mm']:.2f} |"
        )
    return "\n".join(lines)


def generate(results: dict, robustness_report: dict, out_dir: Path, table_dir: Path) -> list:
    """Generate all figures and tables.  Return a list of created paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    created = []

    # 1. Main MPJPE bar chart across selected models/variants
    if results:
        pf.draw_main_mpjpe_bar(results, out_dir / "main_mpjpe_bar.png")
        created.append(out_dir / "main_mpjpe_bar.png")

    # 2. Per-joint MPJPE for every model that carries the data
    for name, r in results.items():
        if r.get("per_joint_mpjpe") is not None:
            out_path = out_dir / f"per_joint_mpjpe_{name}.png"
            joint_names = (
                pf.SKELETON_JOINT_NAMES_17
                if len(r["per_joint_mpjpe"]) == 17
                else [f"J{i}" for i in range(len(r["per_joint_mpjpe"]))]
            )
            pf.draw_per_joint_mpjpe(
                r["per_joint_mpjpe"],
                out_path,
                joint_names=joint_names,
            )
            created.append(out_path)

    # 3. Robustness grid (noise / occlusion / outliers)
    if robustness_report:
        pf.draw_robustness_grid(robustness_report, out_dir / "robustness_grid.png")
        created.append(out_dir / "robustness_grid.png")

    # 4. Main results tables (Markdown + LaTeX)
    if results:
        pt.write_tables(results, table_dir)
        created.append(table_dir / "main_results.md")
        created.append(table_dir / "main_results.tex")

    # 5. Robustness table
    if robustness_report:
        robustness_md = table_dir / "robustness.md"
        robustness_md.write_text(make_robustness_markdown(robustness_report))
        created.append(robustness_md)

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures and tables from existing result JSONs."
    )
    parser.add_argument("--eval_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--robustness", type=str, default=str(ROBUSTNESS_DEFAULT))
    parser.add_argument("--out_dir", type=str, default=str(FIG_DIR))
    parser.add_argument("--table_dir", type=str, default=str(TABLE_DIR))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    table_dir = Path(args.table_dir)
    robustness_path = Path(args.robustness)

    if not eval_dir.exists():
        print(f"ERROR: eval_dir {eval_dir} does not exist.")
        sys.exit(1)

    results = load_evals(eval_dir, DEFAULT_RESULT_FILES)
    if not results:
        print("WARNING: no evaluation JSONs could be loaded.")

    robustness_report = None
    if robustness_path.exists():
        robustness_report = json.loads(robustness_path.read_text())
    else:
        print(f"WARNING: robustness report not found at {robustness_path}")

    created = generate(results, robustness_report, out_dir, table_dir)

    print("\nGenerated files:")
    for f in created:
        print(f"  {f}")

    print("\nModels included:")
    for name, r in results.items():
        print(
            f"  {name:50s}  MPJPE={r['mpjpe_mm']:.2f} mm  "
            f"PA={r['pa_mpjpe_mm']:.2f} mm  AUC={r['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
