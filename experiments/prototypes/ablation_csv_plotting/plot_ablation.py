"""Plot an ablation study from a CSV file.

Usage
-----
    python experiments/prototypes/ablation_csv_plotting/plot_ablation.py \\
        --input experiments/prototypes/ablation_csv_plotting/ablation_template.csv \\
        --output docs/figures/ablation_mpjpe.png \\
        --metric mpjpe_mm \\
        --baseline "Baseline"

The CSV must contain at least the columns ``experiment`` and the metric
specified by ``--metric`` (default ``mpjpe_mm``). An optional ``<metric>_std_mm``
column is used for error bars.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_ablation_csv(csv_path: str, metric: str):
    """Load ablation rows from a CSV file.

    Returns a list of dicts and a list of metric values. Missing numeric values
    become ``None`` so they can be omitted from the plot.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    if "experiment" not in rows[0]:
        raise ValueError("CSV must contain an 'experiment' column")
    if metric not in rows[0]:
        raise ValueError(f"CSV must contain a '{metric}' column")

    return rows


def to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_ablation(
    rows: list,
    metric: str,
    output_path: str,
    baseline_name: str = None,
    title: str = None,
    figsize: tuple = (8, 5),
    dpi: int = 300,
):
    """Render a horizontal bar chart of the ablation results.

    Parameters
    ----------
    rows: list of dict
        Rows from the ablation CSV.
    metric: str
        Metric column to plot (e.g. ``mpjpe_mm``).
    output_path: str
        Path where the PNG will be saved.
    baseline_name: str, optional
        Name of the baseline experiment. It is highlighted and a vertical
        reference line is drawn at its value.
    title: str, optional
        Plot title. Defaults to the metric name.
    figsize: tuple
        Matplotlib figure size.
    dpi: int
        Output resolution.
    """
    experiments = []
    values = []
    errors = []
    std_col = f"{metric}_std_mm"

    for row in rows:
        name = row.get("experiment", "").strip()
        val = to_float(row.get(metric, ""))
        if val is None:
            continue
        experiments.append(name)
        values.append(val)
        err = to_float(row.get(std_col, "")) if std_col in row else None
        errors.append(err)

    if not values:
        raise ValueError(f"No valid numeric values found for metric '{metric}'")

    # Sort so the best (lowest) metric is at the top.
    sorted_triples = sorted(
        zip(experiments, values, errors),
        key=lambda x: (x[1], x[0]),
    )
    experiments, values, errors = zip(*sorted_triples)
    errors = [e for e in errors]  # keep None entries
    has_errors = any(e is not None for e in errors)

    colors = ["#1f77b4"] * len(values)
    if baseline_name and baseline_name in experiments:
        baseline_idx = experiments.index(baseline_name)
        colors[baseline_idx] = "#d62728"

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = range(len(experiments))

    if has_errors:
        ax.barh(
            y_positions,
            values,
            xerr=errors,
            color=colors,
            edgecolor="black",
            capsize=4,
        )
    else:
        ax.barh(
            y_positions,
            values,
            color=colors,
            edgecolor="black",
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(experiments)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " ").upper(), fontsize=12)
    ax.set_ylabel("Experiment", fontsize=12)
    ax.set_title(title or f"Ablation study: {metric}", fontsize=14, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    if baseline_name and baseline_name in experiments:
        baseline_value = values[experiments.index(baseline_name)]
        ax.axvline(
            baseline_value,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline ({baseline_value:.2f})",
        )
        ax.legend(loc="lower right")

    for i, v in enumerate(values):
        x_pos = v + (max(values) - min(values)) * 0.01 if not has_errors else v
        ax.text(
            x_pos,
            i,
            f"{v:.2f}",
            va="center",
            fontsize=9,
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation plot to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot an ablation study CSV.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--metric", type=str, default="mpjpe_mm", help="Metric column to plot")
    parser.add_argument("--baseline", type=str, default=None, help="Name of the baseline experiment")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--dpi", type=int, default=300, help="Output PNG resolution")
    args = parser.parse_args(argv)

    rows = load_ablation_csv(args.input, args.metric)
    plot_ablation(
        rows,
        args.metric,
        args.output,
        baseline_name=args.baseline,
        title=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
