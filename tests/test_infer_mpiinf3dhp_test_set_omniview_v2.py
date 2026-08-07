"""Smoke test for the MPI-INF-3DHP test-set inference script.

Runs the inference script in smoke mode on synthetic data and checks that the
produced output ``.npz`` contains predictions with the expected shape.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent


def test_infer_mpiinf3dhp_test_set_omniview_v2_smoke(tmp_path: Path):
    out_npz = tmp_path / "infer_smoke.npz"
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "infer_mpiinf3dhp_test_set_omniview_v2.py"),
        "--smoke",
        "--out_npz", str(out_npz),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    assert out_npz.exists(), f"Expected output file {out_npz} was not created"

    data = np.load(out_npz)
    assert "smoke" in data.files

    smoke_pred = data["smoke"]
    assert smoke_pred.ndim == 3
    assert smoke_pred.shape[1] == 17  # joints
    assert smoke_pred.shape[2] == 3   # (x, y, z)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
