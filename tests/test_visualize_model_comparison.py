"""CPU smoke tests for scripts/visualize_model_comparison.py."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def out_dir(tmp_path):
    path = tmp_path / "model_comparison_test"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_smoke_script_runs(out_dir: Path) -> None:
    """Run the script in smoke mode and verify outputs."""
    script = Path(__file__).parent.parent / "scripts" / "visualize_model_comparison.py"
    cmd = [sys.executable, str(script), "--smoke", "--out_dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "per_frame_mpjpe.png").exists()
    assert (out_dir / "per_joint_mpjpe.png").exists()
    assert (out_dir / "error_heatmap.png").exists()


def test_help_does_not_crash() -> None:
    script = Path(__file__).parent.parent / "scripts" / "visualize_model_comparison.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
