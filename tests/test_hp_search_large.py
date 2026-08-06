"""CPU smoke tests for the large-run hyperparameter search script.

These tests avoid GPU training entirely.  They verify that the search space
is sampled correctly, that the generated trainer commands are well-formed, and
that a dry-run search writes the expected report files.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.prototypes.hp_search_large_mpiinf3dhp import (
    SearchSpace,
    TrialConfig,
    default_search_space,
    generate_trials,
    parse_args,
    smoke_search_space,
)


def test_default_search_space_coverage():
    """The default space should cover the large-run knobs."""
    space = default_search_space()
    assert len(space.d) >= 2
    assert len(space.residual_hidden) >= 2
    assert len(space.n_st_layers) >= 2
    assert len(space.lr) >= 2
    assert all(0.0 < w for w in space.pp_loss_weight)
    assert all(w >= 0.0 for w in space.epipolar_loss_weight)


def test_smoke_search_space_is_reasonable():
    """The smoke search space should use smaller values for fast CPU runs."""
    smoke = smoke_search_space()
    assert all(0 < v < 128 for v in smoke.d)
    assert all(0 < v < 512 for v in smoke.residual_hidden)
    assert all(v <= 2 for v in smoke.n_st_layers)
    assert max(smoke.batch_size) <= 8


def test_generate_random_trials():
    """Random mode should produce the requested number of distinct trials."""
    space = default_search_space()
    trials = generate_trials(space, "random", 5, seed=123)
    assert len(trials) == 5
    assert len({t.slug() for t in trials}) == 5
    # Each trial should have all hyperparameters set.
    for t in trials:
        assert t.d > 0
        assert t.residual_hidden > 0
        assert t.n_st_layers > 0
        assert t.lr > 0
        assert t.batch_size > 0
        assert t.train_samples > 0


def test_generate_grid_trials_capped():
    """Grid mode should cap the number of combinations at n_trials."""
    space = SearchSpace(
        d=[32, 64],
        residual_hidden=[64, 128],
        n_st_layers=[1, 2],
        lr=[1e-3],
        pp_loss_weight=[0.1],
        epipolar_loss_weight=[0.0],
        cam_aug_pp=[2.0],
        cam_aug_focal=[0.01],
        batch_size=[4],
        train_samples=[100],
    )
    trials = generate_trials(space, "grid", 3, seed=123)
    assert len(trials) == 3
    # Grid order is shuffled; just ensure IDs are sequential.
    assert [t.trial_id for t in trials] == [0, 1, 2]


def test_build_command_dry_run_does_not_need_data():
    """Command building in dry-run mode should not require data files."""
    import argparse
    from experiments.prototypes.hp_search_large_mpiinf3dhp import build_command

    args = argparse.Namespace(
        full=False,
        data_root="/nonexistent/data/webbridge/mpi_inf_3dhp",
        output_dir="/tmp/hp_search_test",
        epochs=2,
    )
    trial = TrialConfig(
        trial_id=0,
        d=64,
        residual_hidden=128,
        n_st_layers=2,
        lr=1e-3,
        pp_loss_weight=0.2,
        epipolar_loss_weight=0.05,
        cam_aug_pp=5.0,
        cam_aug_focal=0.01,
        batch_size=8,
        train_samples=2000,
    )
    cmd = build_command(trial, args, dry_run=True)
    assert "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py" in cmd[1]
    assert cmd[cmd.index("--d") + 1] == "64"
    assert cmd[cmd.index("--residual_hidden") + 1] == "128"
    assert cmd[cmd.index("--n_st_layers") + 1] == "2"
    assert cmd[cmd.index("--model_type") + 1] == "bayesian_tri_v2"
    assert cmd[cmd.index("--batch_size") + 1] == "8"
    assert "_smoke.npz" in " ".join(cmd)


def test_dry_run_search_writes_report():
    """A full dry-run search should produce a trials.json and a markdown report."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "hp_search_test"
        script = Path(__file__).parent.parent / "experiments" / "prototypes" / "hp_search_large_mpiinf3dhp.py"
        cmd = [
            sys.executable,
            str(script),
            "--mode", "smoke",
            "--n_trials", "2",
            "--epochs", "1",
            "--output_dir", str(output_dir),
            "--dry_run",
            "--seed", "42",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        assert "Trial 001/002" in proc.stdout

        trials_json = output_dir / "trials.json"
        report_md = output_dir / "hp_search_report.md"
        assert trials_json.exists()
        assert report_md.exists()

        results = json.loads(trials_json.read_text())
        assert len(results) == 2
        for r in results:
            assert r["returncode"] == 0
            assert "best_val_mpjpe_mm" in r
            assert r["config"]["d"] > 0


def main():
    test_default_search_space_coverage()
    test_smoke_search_space_is_reasonable()
    test_generate_random_trials()
    test_generate_grid_trials_capped()
    test_build_command_dry_run_does_not_need_data()
    test_dry_run_search_writes_report()
    print("All hp_search_large tests passed on CPU.")


if __name__ == "__main__":
    main()
