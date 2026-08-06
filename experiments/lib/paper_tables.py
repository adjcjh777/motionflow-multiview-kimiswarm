"""Reusable table generators for the MotionFlow-MultiView paper."""

from pathlib import Path
from typing import Dict


def _fmt(x, digits: int = 2):
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def make_main_results_table(results: Dict[str, Dict]) -> str:
    """Return a Markdown table of main results."""
    lines = [
        "| Model | Params | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {_fmt(r.get('n_params', 0))} | "
            f"{_fmt(r.get('mpjpe_mm', 0.0))} | {_fmt(r.get('pa_mpjpe_mm', 0.0))} | "
            f"{_fmt(r.get('pck_50', 0.0))} | {_fmt(r.get('pck_100', 0.0))} | "
            f"{_fmt(r.get('pck_150', 0.0))} | {_fmt(r.get('auc', 0.0))} |"
        )
    return "\n".join(lines)


def make_main_results_latex(results: Dict[str, Dict]) -> str:
    """Return a LaTeX table of main results."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{MPI-INF-3DHP cross-subject results.}",
        r"  \label{tab:main_results}",
        r"  \begin{tabular}{lrrrrrrrr}",
        r"    \toprule",
        r"    Model & Params & MPJPE & PA-MPJPE & PCK@50 & PCK@100 & PCK@150 & AUC \\",
        r"    \midrule",
    ]
    for name, r in results.items():
        lines.append(
            f"    {name} & {r.get('n_params', 0)} & "
            f"{r.get('mpjpe_mm', 0.0):.2f} & {r.get('pa_mpjpe_mm', 0.0):.2f} & "
            f"{r.get('pck_50', 0.0):.3f} & {r.get('pck_100', 0.0):.3f} & "
            f"{r.get('pck_150', 0.0):.3f} & {r.get('auc', 0.0):.3f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_robustness_table(robustness_report: Dict) -> str:
    """Return a Markdown robustness table."""
    lines = [
        "| Perturbation | Level | MPJPE (mm) |",
        "|---|---|---:|",
    ]
    for entry in robustness_report.get("noise", []):
        lines.append(f"| Gaussian noise | {entry['noise_std_px']} px | {entry['mpjpe_mm']:.2f} |")
    for entry in robustness_report.get("occlusion", []):
        lines.append(f"| Joint occlusion | {entry['occlusion_rate']*100:.0f}% | {entry['mpjpe_mm']:.2f} |")
    for entry in robustness_report.get("outliers", []):
        lines.append(f"| 2D outliers | {entry['outlier_rate']*100:.0f}% | {entry['mpjpe_mm']:.2f} |")
    return "\n".join(lines)


def write_tables(results: Dict[str, Dict], out_dir: Path) -> None:
    """Write Markdown and LaTeX main results tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main_results.md").write_text(make_main_results_table(results))
    (out_dir / "main_results.tex").write_text(make_main_results_latex(results))
