"""Machine-readable tracker for the next-iteration action plan.

This module encodes the actions in docs/iter_next_action_plan.md as structured
Python data and provides validation / query utilities. It does not touch GPUs or
real datasets and is safe to run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Priority(Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class Phase(Enum):
    CPU_PREP = "cpu_prep"
    GPU_CONVERGENCE = "gpu_convergence"
    PAPER_PACKAGE = "paper_package"


@dataclass
class ActionItem:
    id: str
    name: str
    priority: Priority
    phase: Phase
    rationale: str
    artifact: str
    validation: str
    success_gate: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"


# Synthesized from docs/iter_next_action_plan.md and docs/next_iteration_plan_swarm.md.
_ACTIONS: List[ActionItem] = [
    # P0
    ActionItem(
        id="P0.1",
        name="Finish and evaluate Bayesian Tri v2 large scale",
        priority=Priority.P0,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Already running; largest capacity attempt to beat 8.75 mm anchor.",
        artifact="scripts/run_bayesian_tri_v2_large_scale_wsl.sh, outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.log",
        validation="Clean MPJPE + PA-MPJPE + robustness matrix.",
        success_gate="MPJPE < 8.75 mm, then run 3-5 seeds and declare new anchor.",
    ),
    ActionItem(
        id="P0.2",
        name="Calibration robustness curriculum v2",
        priority=Priority.P0,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Single biggest robustness weakness: rot/focal/pp perturbations.",
        artifact="experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py",
        validation="eval_perturb_model_mpiinf3dhp.py matrix.",
        success_gate="clean <= 9.6 mm, rot_0.5 < 12 mm, focal_1% < 14 mm.",
        depends_on=["P0.1"],
    ),
    ActionItem(
        id="P0.3",
        name="Visibility-gated fusion v2",
        priority=Priority.P0,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Explicit occlusion head; already CPU-smoked and queued.",
        artifact="motionflow_mv/fusion/visibility_gated_fusion.py, experiments/train_crossview_residual_visibility_v2_mpiinf3dhp.py",
        validation="Clean + occlusion sweep (0/10/30/50%).",
        success_gate="clean <= 9.6 mm, >=10% relative gain at 30% occlusion.",
    ),
    ActionItem(
        id="P0.4",
        name="Variable-view inference and view-dropout training",
        priority=Priority.P0,
        phase=Phase.CPU_PREP,
        rationale="Practical rigs have 2-10 views, not 14.",
        artifact="experiments/eval_variable_views.py, experiments/plot_variable_views.py",
        validation="MPJPE@k for k=2..14 on anchor checkpoint.",
        success_gate="k=14 within 0.5 mm of full-view, k=4 < 20 mm, graceful degradation.",
    ),
    ActionItem(
        id="P0.5",
        name="Unified benchmark protocol and repeated seeds",
        priority=Priority.P0,
        phase=Phase.CPU_PREP,
        rationale="Required for any publishable claim.",
        artifact="motionflow_mv/eval/benchmark_protocol.py, experiments/run_repeated_seeds.py",
        validation="3-5 seeds, manifest JSON per run.",
        success_gate="Mean +- std reported for every anchor candidate.",
    ),
    # P1
    ActionItem(
        id="P1.1",
        name="Spatiotemporal (T x V x J) Transformer",
        priority=Priority.P1,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Potential clean MPJPE < 9 mm; expensive, so after P0.",
        artifact="motionflow_mv/fusion/ray_attention_spatiotemporal_model.py",
        validation="CPU smoke + 20-epoch small run.",
        success_gate="clean < 8.75 mm or >=0.3 mm improvement over anchor.",
        depends_on=["P0.1"],
    ),
    ActionItem(
        id="P1.2",
        name="Cross-dataset WebBridge benchmark",
        priority=Priority.P1,
        phase=Phase.CPU_PREP,
        rationale="Paper needs MPI/H36M/AIST/Shelf/Campus tables.",
        artifact="experiments/run_webbridge_benchmark.py",
        validation="Per-dataset MPJPE/PA table.",
        success_gate="Diagnose and fix H36M 101 mm regression.",
    ),
    ActionItem(
        id="P1.3",
        name="Self-supervised masked-view pre-training",
        priority=Priority.P1,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Data-efficiency narrative; low risk if kept additive.",
        artifact="motionflow_mv/data/ssl_dataset.py, experiments/pretrain_ray_attention_ssl.py",
        validation="Pre-train -> fine-tune curve vs. supervised baseline.",
        success_gate=">=10% data reduction at same MPJPE.",
    ),
    ActionItem(
        id="P1.4",
        name="Uncertainty-aware per-view weighting",
        priority=Priority.P1,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Interpretable confidence fusion; complements visibility gating.",
        artifact="motionflow_mv/fusion/ray_attention_temporal_uncertainty_model.py",
        validation="Smoke + small ablation.",
        success_gate="clean <= 9.0 mm or visible robustness matrix improvement.",
    ),
    ActionItem(
        id="P1.5",
        name="Temporal consistency / longer clips",
        priority=Priority.P1,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Reduce jitter; useful if velocity metric is poor.",
        artifact="motionflow_mv/losses/temporal_consistency.py",
        validation="velocity MPJPE on 25-frame clips.",
        success_gate="velocity MPJPE reduced >=5%.",
    ),
    ActionItem(
        id="P1.6",
        name="Graph joint relation v2",
        priority=Priority.P1,
        phase=Phase.GPU_CONVERGENCE,
        rationale="Helps H36M; revisit after cross-dataset fix.",
        artifact="motionflow_mv/fusion/ray_attention_temporal_crossview_residual_graph_joint_model.py",
        validation="H36M S5/Act2 MPJPE.",
        success_gate="H36M < 0.8 mm.",
        depends_on=["P1.2"],
    ),
    # P2
    ActionItem(
        id="P2.1",
        name="Real-time inference optimization",
        priority=Priority.P2,
        phase=Phase.PAPER_PACKAGE,
        rationale="Latency/throughput numbers for paper.",
        artifact="experiments/benchmark_runtime.py",
        validation="FPS on RTX 4090, batch=1.",
        success_gate=">=30 FPS with MPJPE < 9 mm.",
    ),
    ActionItem(
        id="P2.2",
        name="Multi-person multi-view association",
        priority=Priority.P2,
        phase=Phase.PAPER_PACKAGE,
        rationale="System extension; new application scenario.",
        artifact="experiments/associate_multi_person_synthetic.py",
        validation="Synthetic 2-person clip.",
        success_gate="IDF1 > 0.90.",
    ),
    ActionItem(
        id="P2.3",
        name="Action-conditional fusion",
        priority=Priority.P2,
        phase=Phase.PAPER_PACKAGE,
        rationale="Per-action error reduction on H36M.",
        artifact="experiments/ablate_action_aware.py",
        validation="Per-action MPJPE.",
        success_gate=">=2 joints improved on worst action class.",
    ),
    ActionItem(
        id="P2.4",
        name="Gaussian splatting pose regularizer",
        priority=Priority.P2,
        phase=Phase.PAPER_PACKAGE,
        rationale="Novel auxiliary signal; risky.",
        artifact="experiments/test_gaussian_splatting_pose_loss.py",
        validation="Isolated smoke.",
        success_gate="No catastrophic regression on 2-epoch smoke.",
    ),
]


class Plan:
    def __init__(self, actions: List[ActionItem]):
        self.actions = list(actions)

    def by_priority(self, priority: Priority | str) -> List[ActionItem]:
        p = priority if isinstance(priority, Priority) else Priority(priority)
        return [a for a in self.actions if a.priority == p]

    def by_phase(self, phase: Phase | str) -> List[ActionItem]:
        ph = phase if isinstance(phase, Phase) else Phase(phase)
        return [a for a in self.actions if a.phase == ph]

    def get(self, action_id: str) -> ActionItem:
        for a in self.actions:
            if a.id == action_id:
                return a
        raise KeyError(action_id)

    def validate(self) -> List[str]:
        """Return a list of validation errors; empty list means valid."""
        errors: List[str] = []
        ids = {a.id for a in self.actions}
        if len(ids) != len(self.actions):
            errors.append("Duplicate action IDs found.")
        for a in self.actions:
            if not a.id:
                errors.append("Action missing id.")
            if not a.name or not a.name.strip():
                errors.append(f"{a.id}: missing name.")
            if not a.artifact or not a.artifact.strip():
                errors.append(f"{a.id}: missing artifact.")
            if not a.success_gate or not a.success_gate.strip():
                errors.append(f"{a.id}: missing success_gate.")
            for dep in a.depends_on:
                if dep not in ids:
                    errors.append(f"{a.id}: unknown dependency {dep}.")
        return errors


def load_plan() -> Plan:
    """Load the synthesized next-iteration plan."""
    return Plan(_ACTIONS)


def main() -> None:
    plan = load_plan()
    errors = plan.validate()
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("Next-iteration action plan")
    print(f"Total actions: {len(plan.actions)}")
    for p in Priority:
        actions = plan.by_priority(p)
        print(f"  {p.value}: {len(actions)} actions")
    print()
    print("P0 actions:")
    for a in plan.by_priority(Priority.P0):
        print(f"  {a.id} {a.name}")
        print(f"      artifact: {a.artifact}")
        print(f"      gate: {a.success_gate}")


if __name__ == "__main__":
    main()
