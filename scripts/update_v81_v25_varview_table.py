#!/usr/bin/env python3
"""Wait for v81 and v25 stability variable-view JSONs and update the docs table."""
import json
import time
from pathlib import Path

A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
V81_JSON = Path(f"{A800_REPO}/outputs/variable_view_v81_true_gt_medium_a800.json")
V25_JSON = Path(f"{A800_REPO}/outputs/variable_view_v25_true_gt_stability_a800.json")
DOCS_PATH = Path("docs/paper_sparse_view_section_20260811.md")


def load_per_dataset(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    return data.get("per_dataset", data)


def compute_combined(per_dataset: dict):
    """Return dict k -> combined S9+S11 MPJPE (simple mean)."""
    combined = {}
    keys = sorted([int(k) for k in per_dataset["S9"].keys()])
    for k in keys:
        s9 = per_dataset["S9"][str(k)]["mpjpe_at_k"]
        s11 = per_dataset["S11"][str(k)]["mpjpe_at_k"]
        combined[k] = round((s9 + s11) / 2.0, 2)
    return combined


def make_table(v81_combined: dict, v25_combined: dict):
    lines = [
        "### v81 vs. v25 stability (true-GT H36M variable-view)",
        "",
        "Comparison of the v81 temporal-pose-attention medium checkpoint and the v25 stability checkpoint on the same S9+S11 true-GT H36M variable-view protocol (all C(4,k) view subsets per `k`, `clip_len = 13`).  Combined values are the simple mean of per-subject MPJPE@k.",
        "",
        "| Variant | k=2 (mm) | k=3 (mm) | k=4 (mm) | Full-view test (mm) | Source |",
        "|---|---:|---:|---:|---:|---|",
        f"| v81 (temporal-pose-attention) | **{v81_combined[2]:.2f}** | **{v81_combined[3]:.2f}** | **{v81_combined[4]:.2f}** | 37.83 | `outputs/variable_view_v81_true_gt_medium_a800.json` |",
        f"| v25 stability | **{v25_combined[2]:.2f}** | **{v25_combined[3]:.2f}** | **{v25_combined[4]:.2f}** | 31.56 | `outputs/variable_view_v25_true_gt_stability_a800.json` |",
        "",
    ]
    return "\n".join(lines) + "\n"


def update_docs(v81_combined: dict, v25_combined: dict):
    text = DOCS_PATH.read_text()
    marker = "## 2. H36M true-GT variable-view results\n\n"
    if marker not in text:
        raise RuntimeError("Could not find H36M true-GT variable-view section marker")

    table_text = make_table(v81_combined, v25_combined)

    # Insert before the first subsection after the existing table (before "### Observations")
    insert_marker = "### Observations\n"
    if insert_marker in text:
        text = text.replace(insert_marker, table_text + insert_marker, 1)
    else:
        # append after marker
        idx = text.find(marker) + len(marker)
        text = text[:idx] + "\n" + table_text + text[idx:]

    DOCS_PATH.write_text(text)
    print(f"Updated {DOCS_PATH}")


def main(timeout_seconds: int = 3600, poll_interval: int = 60):
    start = time.time()
    while time.time() - start < timeout_seconds:
        if V81_JSON.exists() and V25_JSON.exists():
            print("Both JSONs found, computing table...")
            v81 = compute_combined(load_per_dataset(V81_JSON))
            v25 = compute_combined(load_per_dataset(V25_JSON))
            print("v81 combined:", v81)
            print("v25 stability combined:", v25)
            update_docs(v81, v25)
            return
        print(f"Waiting... v81={V81_JSON.exists()}, v25={V25_JSON.exists()}")
        time.sleep(poll_interval)
    raise TimeoutError("Variable-view JSONs did not appear within timeout")


if __name__ == "__main__":
    main(timeout_seconds=14400, poll_interval=60)
