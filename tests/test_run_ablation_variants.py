"""Unit tests for scripts/run_ablation_variants.py"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_ablation_variants import (
    _build_command,
    _merge_args,
    _validate_config,
)


def test_build_command_with_bools_and_values():
    args = {
        "--use_flag": True,
        "--skip_flag": False,
        "--epochs": 2,
        "--lr": 1e-3,
        "--manifest": ["a.yaml", "b.yaml"],
    }
    cmd = _build_command("train.py", args)
    assert cmd == [
        sys.executable,
        "train.py",
        "--use_flag",
        "--epochs",
        "2",
        "--lr",
        "0.001",
        "--manifest",
        "a.yaml b.yaml",
    ]


def test_merge_args_override():
    base = {"--epochs": 10, "--batch_size": 4}
    override = {"--epochs": 2, "--new_flag": True}
    merged = _merge_args(base, override)
    assert merged == {"--epochs": 2, "--batch_size": 4, "--new_flag": True}


def test_validate_config_ok(tmp_path):
    cfg = {
        "base": {"script": "train.py"},
        "variants": [
            {"name": "v1"},
            {"name": "v2"},
        ],
    }
    _validate_config(cfg)


def test_validate_config_missing_base():
    cfg = {"variants": [{"name": "v1"}]}
    with pytest.raises(ValueError, match="missing required 'base' section"):
        _validate_config(cfg)


def test_validate_config_missing_script():
    cfg = {"base": {}, "variants": [{"name": "v1"}]}
    with pytest.raises(ValueError, match="base missing required keys"):
        _validate_config(cfg)


def test_validate_config_no_variants():
    cfg = {"base": {"script": "train.py"}}
    with pytest.raises(ValueError, match="at least one variant"):
        _validate_config(cfg)


def test_validate_config_duplicate_names():
    cfg = {
        "base": {"script": "train.py"},
        "variants": [{"name": "v1"}, {"name": "v1"}],
    }
    with pytest.raises(ValueError, match="Duplicate variant names"):
        _validate_config(cfg)


class TestRunnerScript:
    """Tests that invoke the runner script via subprocess (integration)."""

    def test_dry_run_example_config(self, tmp_path):
        import subprocess

        repo = Path(__file__).parent.parent
        config = repo / "configs" / "ablations" / "example_kap_ba_sweep.yaml"

        result = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "run_ablation_variants.py"),
                "--config",
                str(config),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo)},
        )

        assert result.returncode == 0, result.stderr
        assert "v23_kap_no_ba" in result.stdout
        assert "v24_kap_fixed_ba" in result.stdout
        assert "--use_neural_bundle_adjustment_v21" in result.stdout

        # Summary JSON should be written.
        output_dir = repo / "outputs" / "ablations" / "example_kap_ba_sweep"
        summary_path = output_dir / "ablation_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_variants"] == 2
        assert summary["is_dry_run"] is True
        assert summary["dry_run"] == 2
