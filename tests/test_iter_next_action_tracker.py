"""CPU smoke test for the next-iteration action-plan tracker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "prototypes"))

from iter_next_action_tracker import load_plan, Priority, Phase


def test_plan_loads_and_validates():
    plan = load_plan()
    errors = plan.validate()
    assert not errors, f"Plan validation failed: {errors}"


def test_priorities_and_phases():
    plan = load_plan()
    for item in plan.actions:
        assert item.priority in Priority
        assert item.phase in Phase


def test_action_counts():
    plan = load_plan()
    p0 = plan.by_priority(Priority.P0)
    p1 = plan.by_priority(Priority.P1)
    p2 = plan.by_priority(Priority.P2)
    assert len(p0) == 5
    assert len(p1) == 6
    assert len(p2) == 4
    assert len(plan.actions) == 15


def test_dependencies_resolve():
    plan = load_plan()
    ids = {a.id for a in plan.actions}
    for item in plan.actions:
        for dep in item.depends_on:
            assert dep in ids, f"{item.id} depends on unknown {dep}"


def test_p0_cpu_prep_items_have_cpu_artifacts():
    plan = load_plan()
    for item in plan.by_phase(Phase.CPU_PREP):
        assert item.artifact, f"{item.id} missing artifact"


if __name__ == "__main__":
    test_plan_loads_and_validates()
    test_priorities_and_phases()
    test_action_counts()
    test_dependencies_resolve()
    test_p0_cpu_prep_items_have_cpu_artifacts()
    print("All action-plan tracker smoke tests passed.")
