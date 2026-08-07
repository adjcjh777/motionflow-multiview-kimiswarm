"""CPU smoke test for the v4 WebBridge multi-dataset trainer.

Runs ``experiments/train_omniview_fusion_v4_webbridge_multi.py --smoke`` and
asserts that it completes without error.  This test is intentionally lightweight
and does not require external data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_omniview_fusion_v4_smoke():
    """Trainer --smoke flag runs end-to-end on CPU."""
    script = Path(__file__).parent.parent / "experiments" / "train_omniview_fusion_v4_webbridge_multi.py"
    result = subprocess.run(
        [sys.executable, str(script), "--smoke"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Smoke test failed:\n{result.stderr}"
    assert "Best val MPJPE" in result.stdout, "Expected final MPJPE log not found"
