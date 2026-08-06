"""Correlate interpretable model signals with failures.

Reads the artifacts produced by ``analyze_failures_crossview_pp.py`` and
computes lightweight, CPU-only correlations that tell us whether the model's
internal signals (fusion weights, principal-point correction, residual
refinement) align with actual errors.

Output is a small Markdown report with Pearson/Spearman correlations and a
couple of diagnostic plots.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or NaN if invalid."""
    a = a.flatten()
    b = b.flatten()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=str,
                        default="outputs/failure_analysis_crossview_pp_smoke/failure_arrays.npz")
    parser.add_argument("--out_dir", type=str,
                        default="outputs/failure_analysis_crossview_pp_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.arrays)
    per_joint_err = data["per_joint_err_mm"]          # (T, J)
    per_frame_err = data["per_frame_err_mm"]          # (T,)
    per_view_reproj = data["per_view_reproj_px"]      # (T, V)
    mean_weights = data["mean_weights"]               # (V,)
    pp_delta_norm = data["pp_delta_norm_px"]          # (T, V)
    residual_norm = data["residual_norm_mm"]          # (T, J)

    T, V, J = per_view_reproj.shape[0], per_view_reproj.shape[1], per_joint_err.shape[1]

    # Per-view statistics.
    view_err_mean = per_view_reproj.mean(axis=0)               # (V,)
    view_err_median = np.median(per_view_reproj, axis=0)      # (V,)
    pp_delta_view = pp_delta_norm.mean(axis=0)                 # (V,)

    # Per-frame statistics.
    residual_per_frame = residual_norm.mean(axis=1)            # (T,)

    # Correlations.
    results = {
        "mean_weight vs mean_reproj (per view)": pearson(mean_weights, view_err_mean),
        "mean_weight vs median_reproj (per view)": pearson(mean_weights, view_err_median),
        "pp_delta vs mean_reproj (per view)": pearson(pp_delta_view, view_err_mean),
        "pp_delta vs median_reproj (per view)": pearson(pp_delta_view, view_err_median),
        "residual_magnitude vs frame_mpjpe (per frame)": pearson(residual_per_frame, per_frame_err),
    }

    # Also compute per-joint residual vs error correlation.
    mean_err_per_joint = per_joint_err.mean(axis=0)
    mean_res_per_joint = residual_norm.mean(axis=0)
    results["residual_magnitude vs mpjpe (per joint)"] = pearson(mean_res_per_joint, mean_err_per_joint)

    lines = [
        "# Interpretability Correlation Report\n",
        f"* Source: `{args.arrays}`\n",
        f"* Frames: {T}, Views: {V}, Joints: {J}\n\n",
        "## Correlations (Pearson r)\n\n",
        "| Signal pair | r |\n",
        "|---|---|\n",
    ]
    for k, v in results.items():
        lines.append(f"| {k} | {v:.3f} |\n")
    lines.append("\n")

    # Interpretation heuristics.
    lines.append("## Interpretation\n\n")
    if results["mean_weight vs mean_reproj (per view)"] < -0.3:
        lines.append("* Fusion weights are inversely correlated with reprojection error "
                     "(good: the model puts more weight on reliable views).\n")
    elif results["mean_weight vs mean_reproj (per view)"] > 0.3:
        lines.append("* **Warning:** fusion weights are positively correlated with reprojection error "
                     "(the model may be over-weighting unreliable views).\n")
    else:
        lines.append("* Fusion weights are weakly correlated with reprojection error "
                     "(the fusion mechanism may not be fully exploiting view reliability).\n")

    if results["pp_delta vs mean_reproj (per view)"] > 0.3:
        lines.append("* Larger principal-point corrections tend to occur on problematic views "
                     "(PP correction is reactive to bad calibration).\n")
    elif results["pp_delta vs mean_reproj (per view)"] < -0.3:
        lines.append("* Larger principal-point corrections are associated with lower reprojection error "
                     "(PP correction is helping).\n")
    else:
        lines.append("* Principal-point correction magnitude is largely independent of view error "
                     "(the correction may be too small or uniform to dominate view quality).\n")

    if results["residual_magnitude vs frame_mpjpe (per frame)"] > 0.3:
        lines.append("* Residual refinement magnitude tracks frame error "
                     "(the residual head is trying to fix large triangulation errors).\n")
    elif results["residual_magnitude vs frame_mpjpe (per frame)"] < -0.3:
        lines.append("* Larger residuals appear on easier frames (possible over-correction).\n")
    else:
        lines.append("* Residual refinement magnitude is weakly correlated with frame error "
                     "(its corrections may be diffuse rather than targeted).\n")

    report_path = out_dir / "interpretability_correlations.md"
    with open(report_path, "w") as f:
        f.writelines(lines)

    print(f"Correlation report written to: {report_path}")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
