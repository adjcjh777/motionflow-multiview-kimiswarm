"""CPU smoke test for scripts/visualize_variable_view_failure.py.

Ensures the diagnostic script runs end-to-end with a freshly initialised v2 model
and produces the expected JSON summaries and figure files.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def out_dir(tmp_path):
    path = tmp_path / "failure_analysis_variable_views_test"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_smoke_script_runs(out_dir: Path) -> None:
    """Run the script in smoke mode and verify outputs."""
    script = Path(__file__).parent.parent / "scripts" / "visualize_variable_view_failure.py"
    cmd = [sys.executable, str(script), "--smoke", "--out_dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    for k in [2, 3, 4]:
        assert (out_dir / f"summary_k{k}.json").exists()
        assert (out_dir / f"per_joint_error_k{k}.png").exists()
        assert (out_dir / f"view_weights_k{k}.png").exists()
        assert (out_dir / f"visibility_k{k}.png").exists()
        assert (out_dir / f"triangulation_residual_k{k}.png").exists()
        assert (out_dir / f"residual_distribution_k{k}.png").exists()

    assert (out_dir / "summary.json").exists()


def test_help_does_not_crash() -> None:
    script = Path(__file__).parent.parent / "scripts" / "visualize_variable_view_failure.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
