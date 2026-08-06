"""CPU smoke tests for motionflow_mv.eval.benchmark_protocol."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from motionflow_mv.eval import BenchmarkConfig, BenchmarkProtocol


class DummyModel(torch.nn.Module):
    """Trivial model that returns zeros shaped like the ground-truth pose."""

    def forward(self, x, K=None, R=None, t=None):
        # x: (B, T, V, J, C) -> return (B, T, J, 3)
        batch, t_len, _, joints, _ = x.shape
        return torch.zeros(batch, t_len, joints, 3, dtype=x.dtype, device=x.device)


def _make_dataloader(batch_size=2, n_batches=2, clip_len=5, n_views=3, n_joints=17):
    """Build a deterministic DataLoader for smoke tests."""
    n = batch_size * n_batches
    x = torch.randn(n, clip_len, n_views, n_joints, 3)
    y = torch.zeros(n, clip_len, n_joints, 3)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(n, n_views, 1, 1)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(n, n_views, 1, 1)
    t = torch.zeros(n, n_views, 3)
    ds = TensorDataset(x, y, K, R, t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_evaluate_model_returns_metrics():
    cfg = BenchmarkConfig(
        dataset="mpiinf3dhp",
        split="test",
        clip_len=5,
        unit_scale=1.0,
    )
    protocol = BenchmarkProtocol(cfg)
    model = DummyModel()
    dataloader = _make_dataloader()

    parents = np.full(17, -1, dtype=np.int64)
    report = protocol.evaluate_model(model, dataloader, device="cpu", parents=parents)

    assert "mpjpe" in report
    assert "pa_mpjpe" in report
    assert "root_rel_mpjpe" in report
    assert "velocity_mpjpe" in report
    assert "bone_length_error" in report
    assert report["mpjpe"] == pytest.approx(0.0, abs=1e-5)
    assert report["root_joint"] == cfg.root_joint
    assert report["unit_scale"] == cfg.unit_scale


def test_run_persists_results(tmp_path):
    cfg = BenchmarkConfig(dataset="mpiinf3dhp", split="test", clip_len=5, unit_scale=1.0)
    protocol = BenchmarkProtocol(cfg)
    model = DummyModel()
    dataloader = _make_dataloader(batch_size=2, n_batches=1)

    out_dir = tmp_path / "results"
    report = protocol.run(model, dataloader, device="cpu", out_dir=out_dir)

    results_path = out_dir / "results.json"
    summary_path = out_dir / "results.txt"
    assert results_path.exists()
    assert summary_path.exists()
    assert "mpjpe" in report


def test_train_runs_trivial_script(tmp_path):
    script = tmp_path / "dummy_train.py"
    script.write_text(
        """
import sys
from pathlib import Path

# Minimal parser: --output <path> --seed <seed>
args = sys.argv[1:]
out = args[args.index("--output") + 1]
Path(out).write_text(\"ok\")
"""
    )
    protocol = BenchmarkProtocol(BenchmarkConfig(dataset="h36m", split="test"))
    out_file = tmp_path / "checkpoint.pth"
    entry = protocol.train(
        script=script,
        base_args=["--lr", "1e-3"],
        seed=42,
        output=out_file,
        dry_run=False,
    )
    assert entry["status"] == "completed"
    assert out_file.exists()


def test_run_multi_seed_dry_run(tmp_path):
    protocol = BenchmarkProtocol(BenchmarkConfig(dataset="mpiinf3dhp", split="test"))
    manifest = protocol.run_multi_seed(
        script=tmp_path / "dummy_train.py",
        base_args="--epochs 1",
        seeds=[1, 2, 3],
        out_dir=tmp_path / "seeds",
        base_name="model",
        dry_run=True,
    )
    assert manifest["seeds"] == [1, 2, 3]
    assert len(manifest["checkpoints"]) == 3
    for seed, entry in manifest["checkpoints"].items():
        assert entry["status"] == "dry_run"
        assert int(seed) in [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
