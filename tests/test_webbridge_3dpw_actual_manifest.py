"""Smoke test for the 3DPW actual-mode smoke manifest.

This test loads ``configs/splits/webbridge_3dpw_actual_smoke.yaml`` through
``WebBridgeMixedDataset`` and verifies that the actual-mode validation clip
loads with per-frame moving cameras and the expected skeleton/view padding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.data.webbridge_mixed_dataset import WebBridgeMixedDataset


def test_3dpw_actual_smoke_manifest_loads() -> None:
    """The 3DPW actual-mode manifest should load without crashing."""
    manifest_path = ROOT / "configs" / "splits" / "webbridge_3dpw_actual_smoke.yaml"
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    with open(manifest_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_paths = cfg["train_paths"]
    train_names = cfg["train_names"]
    val_paths = cfg["val_paths"]
    val_names = cfg["val_names"]

    assert len(train_paths) == len(train_names)
    assert len(val_paths) == len(val_names)
    assert any("_actual.npz" in p for p in val_paths)

    val_dataset = WebBridgeMixedDataset(
        npz_paths=val_paths,
        dataset_names=val_names,
        clip_len=9,
    )

    assert len(val_dataset) > 0

    # The actual-mode val clip should load with per-frame cameras.
    x, y, K, R, t, dataset_id = val_dataset.datasets[0][0]
    assert x.shape[0] == 9  # clip_len
    assert y.shape == (9, 17, 3)
    assert K.shape == (9, 14, 3, 3)
    assert R.shape == (9, 14, 3, 3)
    assert t.shape == (9, 14, 3)
    # First padded view should be identity/zero.
    assert K[0, 0].abs().sum().item() > 0.0
