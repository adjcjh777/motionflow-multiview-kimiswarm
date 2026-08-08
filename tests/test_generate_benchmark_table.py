"""CPU smoke tests for scripts/generate_benchmark_table.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_benchmark_table import (
    _build_markdown,
    _collect_rows,
    _extract_rows,
    _select_metrics,
    main,
)


def _make_webbridge_json(tmp_path: Path) -> Path:
    data = {
        "manifest": "configs/benchmark_webbridge_crossview_residual_smoke.yaml",
        "model_config": {"model": "crossview_residual", "checkpoint": "outputs/crossview_residual.pth"},
        "results": [
            {
                "dataset": "mpi_s2_seq1_v14",
                "mpjpe_mm": 14.71,
                "pa_mpjpe_mm": 13.86,
                "pck_50": 0.9972,
                "pck_100": 1.0,
                "pck_150": 1.0,
                "pck_auc": 0.9019,
            }
        ],
    }
    path = tmp_path / "webbridge.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_auto_eval_json(tmp_path: Path) -> Path:
    data = {
        "checkpoint": "outputs/v23_kap_no_ba.pth",
        "model": "omniview_fusion_v5",
        "dry_run": True,
        "datasets": [
            {
                "name": "h36m",
                "sequences": [
                    {
                        "name": "h36m__s_11_acts_02_multiview_m",
                        "metrics": {
                            "mpjpe": 20.24,
                            "pa_mpjpe": 18.11,
                            "pck@50mm": 0.9993,
                            "pck@100mm": 1.0,
                            "pck@150mm": 1.0,
                            "pck_auc": 0.9936,
                        },
                        "status": "dry_run",
                    }
                ],
                "metrics": {
                    "mpjpe": 20.24,
                    "pa_mpjpe": 18.11,
                    "pck@50mm": 0.9993,
                    "pck@100mm": 1.0,
                    "pck@150mm": 1.0,
                    "pck_auc": 0.9936,
                },
            }
        ],
        "summary": {"per_dataset": {}, "overall": {}},
    }
    path = tmp_path / "auto_eval.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_extract_rows_webbridge(tmp_path):
    path = _make_webbridge_json(tmp_path)
    data = json.loads(path.read_text())
    rows = _extract_rows(data, path)
    assert len(rows) == 1
    assert rows[0]["model"] == "crossview_residual"
    assert rows[0]["dataset"] == "mpi_s2_seq1_v14"
    assert pytest.approx(rows[0]["mpjpe"]) == 14.71


def test_extract_rows_auto_eval(tmp_path):
    path = _make_auto_eval_json(tmp_path)
    data = json.loads(path.read_text())
    rows = _extract_rows(data, path)
    assert len(rows) == 1
    assert rows[0]["model"] == "omniview_fusion_v5"
    assert rows[0]["dataset"] == "h36m"
    assert rows[0]["sequence"] == "h36m__s_11_acts_02_multiview_m"
    assert pytest.approx(rows[0]["mpjpe"]) == 20.24


def test_build_markdown_contains_headers():
    rows = [
        {
            "model": "m",
            "dataset": "d",
            "sequence": "s",
            "mpjpe": 10.0,
            "pa_mpjpe": 8.0,
        }
    ]
    metrics = _select_metrics(rows, ["mpjpe", "pa_mpjpe"])
    md = _build_markdown(rows, metrics)
    assert "Model" in md
    assert "Dataset" in md
    assert "MPJPE (mm)" in md
    assert "PA-MPJPE (mm)" in md


def test_collect_rows(tmp_path):
    p1 = _make_webbridge_json(tmp_path)
    p2 = _make_auto_eval_json(tmp_path)
    rows = _collect_rows([p1, p2])
    assert len(rows) == 2


def test_main_writes_markdown(tmp_path, capsys):
    path = _make_webbridge_json(tmp_path)
    out = tmp_path / "table.md"
    main(["--inputs", str(path), "--output", str(out)])
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "crossview_residual" in text
    assert "mpi_s2_seq1_v14" in text
    assert "14.71" in text


def test_main_writes_csv(tmp_path, capsys):
    path = _make_auto_eval_json(tmp_path)
    out_md = tmp_path / "table.md"
    out_csv = tmp_path / "table.csv"
    main(["--inputs", str(path), "--output", str(out_md), "--csv", str(out_csv)])
    assert out_csv.exists()
    text = out_csv.read_text(encoding="utf-8")
    assert "h36m" in text
    assert "20.24" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
