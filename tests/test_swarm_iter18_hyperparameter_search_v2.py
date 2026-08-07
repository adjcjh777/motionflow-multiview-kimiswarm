"""CPU smoke tests for the swarm-iter18 hyperparameter search harness.

These tests do not run any trainer; they only exercise the orchestration,
search-space sampling, ASHA rung generation, and command building.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import experiments.prototypes.swarm_iter18.hyperparameter_search_v2 as hp


def test_asha_rung_epochs():
    assert hp.asha_rung_epochs(50, 4) == [3, 12, 28, 50]
    assert hp.asha_rung_epochs(10, 1) == [10]
    assert hp.asha_rung_epochs(8, 3) == [1, 3, 8]


def test_search_space_sampling():
    space = hp.smoke_search_space()
    import random
    rng = random.Random(0)
    cfg = space.sample_random(rng)
    assert "d" in cfg
    assert "uncertainty_loss_weight" in cfg
    assert cfg["d"] in space.d


def test_generate_trials_random():
    space = hp.smoke_search_space()
    trials = hp.generate_trials(space, "random", 5, seed=0)
    assert len(trials) == 5
    ids = [t.trial_id for t in trials]
    assert ids == list(range(5))


def test_generate_trials_grid():
    space = hp.smoke_search_space()
    # Force a tiny grid by overriding all attributes to single values.
    for key in space.__annotations__:
        setattr(space, key, [getattr(space, key)[0]])
    trials = hp.generate_trials(space, "grid", 2, seed=0)
    assert len(trials) == 1  # only one combination


def test_build_command_omniview_flag_filtering():
    """Omni-only flags should only appear when --omniview is set."""
    space = hp.smoke_search_space(omniview=False)
    trials = hp.generate_trials(space, "random", 1, seed=0)
    trial = trials[0]
    args = hp.parse_args([
        "--mode", "smoke", "--dry_run", "--output_dir", "tmp/hp_test"
    ])
    output = Path("tmp") / "hp_test" / "test.pth"
    cmd = hp.build_command(trial, args, epochs=1, output_path=output, dry_run=True)
    cmd_str = " ".join(cmd)
    assert "--uncertainty_loss_weight" not in cmd_str
    assert "--bone_loss_weight" not in cmd_str
    assert "--graph_attention_heads" not in cmd_str

    args_omni = hp.parse_args([
        "--mode", "smoke", "--dry_run", "--omniview", "--output_dir", "tmp/hp_test_omni"
    ])
    cmd_omni = hp.build_command(trial, args_omni, epochs=1, output_path=output, dry_run=True)
    cmd_omni_str = " ".join(cmd_omni)
    assert "--uncertainty_loss_weight" in cmd_omni_str
    assert "--bone_loss_weight" in cmd_omni_str
    assert "--graph_attention_heads" in cmd_omni_str


def test_default_search_space_omniview_difference():
    legacy = hp.default_search_space(omniview=False)
    omni = hp.default_search_space(omniview=True)
    assert len(legacy.uncertainty_loss_weight) == 1
    assert len(omni.uncertainty_loss_weight) > 1
    assert len(legacy.bone_loss_weight) == 1
    assert len(omni.bone_loss_weight) > 1


def test_skip_existing_flag_present():
    args = hp.parse_args([
        "--mode", "smoke", "--dry_run", "--skip_existing",
    ])
    assert args.skip_existing is True
