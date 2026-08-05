"""Generate a Markdown results table from one or more JSON metric files.

Usage
-----
    python experiments/generate_results_markdown.py \
        --inputs outputs/eval_baseline.json outputs/eval_pp.json \
        --labels Baseline PP \
        --output docs/results_generated.md
"""

import argparse
import json
from pathlib import Path


def load_metrics(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    # Flatten nested dicts if present.
    if "clean" in data and isinstance(data["clean"], dict):
        return data["clean"]
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown results table from JSON metrics")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="JSON files to compare")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Label for each JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output Markdown file")
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("Number of inputs and labels must match")

    metrics = ["mpjpe", "pa_mpjpe", "pck@50mm", "pck@100mm", "pck@150mm", "pck_auc"]
    lines = ["| Model | " + " | ".join(m for m in metrics) + " |", "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"]

    for label, path in zip(args.labels, args.inputs):
        data = load_metrics(path)
        row = [label]
        for m in metrics:
            val = data.get(m)
            if val is None:
                row.append("—")
            else:
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(row) + " |")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote Markdown table to {out_path}")


if __name__ == "__main__":
    main()
