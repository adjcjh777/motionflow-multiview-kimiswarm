"""CPU smoke test for the ablation CSV template and plotting script."""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.prototypes.ablation_csv_plotting.plot_ablation import (
    load_ablation_csv,
    plot_ablation,
)


def _write_sample_csv(csv_path: Path):
    fieldnames = [
        "experiment",
        "component",
        "config",
        "mpjpe_mm",
        "mpjpe_std_mm",
        "pa_mpjpe_mm",
        "pck_100",
        "n_params",
        "notes",
    ]
    rows = [
        {
            "experiment": "Baseline",
            "component": "full_model",
            "config": "d=128;rh=256",
            "mpjpe_mm": "8.52",
            "mpjpe_std_mm": "0.03",
            "pa_mpjpe_mm": "6.11",
            "pck_100": "0.99",
            "n_params": "1244000",
            "notes": "anchor run",
        },
        {
            "experiment": "No temporal",
            "component": "temporal",
            "config": "d=128;rh=256",
            "mpjpe_mm": "9.31",
            "mpjpe_std_mm": "0.05",
            "pa_mpjpe_mm": "6.80",
            "pck_100": "0.98",
            "n_params": "1180000",
            "notes": "remove temporal layers",
        },
        {
            "experiment": "No cross-view",
            "component": "cross_view",
            "config": "d=128;rh=256",
            "mpjpe_mm": "10.14",
            "mpjpe_std_mm": "0.04",
            "pa_mpjpe_mm": "7.25",
            "pck_100": "0.97",
            "n_params": "1050000",
            "notes": "remove cross-view attention",
        },
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_template_csv_exists():
    template = (
        Path(__file__).parent.parent
        / "experiments"
        / "prototypes"
        / "ablation_csv_plotting"
        / "ablation_template.csv"
    )
    assert template.exists(), f"Template CSV not found: {template}"
    with open(template, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows, "Template CSV is empty"
    assert "experiment" in reader.fieldnames
    assert "mpjpe_mm" in reader.fieldnames


def test_plot_ablation_creates_png():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "ablation.csv"
        _write_sample_csv(csv_path)
        out_path = Path(tmpdir) / "ablation.png"

        rows = load_ablation_csv(str(csv_path), "mpjpe_mm")
        plot_ablation(
            rows,
            "mpjpe_mm",
            str(out_path),
            baseline_name="Baseline",
            title="Smoke test ablation",
        )
        assert out_path.exists(), "PNG was not created"
        assert out_path.stat().st_size > 0, "PNG is empty"


if __name__ == "__main__":
    test_template_csv_exists()
    test_plot_ablation_creates_png()
    print("Smoke tests passed.")
