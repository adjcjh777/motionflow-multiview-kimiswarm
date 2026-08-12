#!/usr/bin/env python3
"""Validate the GitHub issue and PR drafts produced by the 20-agent swarm.

Usage:
    python docs/swarm_iter_next/design_github_next_steps/validate_drafts.py

Returns 0 if the drafts contain the required sections and reference all
expected subagent deliverables. No training is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ISSUE_DRAFT = REPO_ROOT / "docs" / "swarm_iter_next" / "github_issue_draft.md"
PR_DRAFT = REPO_ROOT / "docs" / "swarm_iter_next" / "github_pr_draft.md"

REQUIRED_ISSUE_SECTIONS = [
    "## Summary",
    "## Current best model",
    "## Swarm deliverables",
    "## Key findings",
    "## Next steps",
    "## Blockers",
    "## Related files",
]

REQUIRED_PR_SECTIONS = [
    "## Summary",
    "## Key changes",
    "## Verified results",
    "## Testing",
    "## Checklist",
    "## Related issues",
]

# A representative subset of swarm deliverables that the drafts must reference.
EXPECTED_DELIVERABLE_PATHS = [
    "docs/swarm_iter_next/implement_robust_triangulation_baseline",
    "docs/swarm_iter_next/design_adaptive_view_selection",
    "docs/swarm_iter_next/design_graph_joint_relation",
    "docs/swarm_iter_next/design_camera_positional_encoding_report.md",
    "docs/swarm_iter_next/design_self_supervised_pretext",
    "docs/swarm_iter_next/design_multi_task_shape_pose",
    "docs/swarm_iter_next/design_a800_benchmark_script",
    "docs/swarm_iter_next/design_docker_reproducibility",
    "docs/swarm_iter_next/design_github_next_steps",
]


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _check_sections(name: str, content: str, sections: list[str]) -> list[str]:
    missing = [s for s in sections if s not in content]
    if missing:
        print(f"[FAIL] {name} missing sections: {missing}")
    else:
        print(f"[OK] {name} contains all required sections")
    return missing


def _check_references(name: str, content: str) -> list[str]:
    """Check that the draft references each expected deliverable and that it exists on disk."""
    missing: list[str] = []
    for path_str in EXPECTED_DELIVERABLE_PATHS:
        if path_str not in content:
            missing.append(f"{path_str} (not mentioned)")
            continue
        deliverable_path = REPO_ROOT / path_str
        if not deliverable_path.exists():
            missing.append(f"{path_str} (missing on disk)")
    if missing:
        print(f"[FAIL] {name} missing references: {missing}")
    else:
        print(f"[OK] {name} references all expected deliverables and they exist on disk")
    return missing


def main() -> int:
    """Run validation and return the number of failures."""
    failures: list[str] = []

    for label, path, sections in [
        ("Issue draft", ISSUE_DRAFT, REQUIRED_ISSUE_SECTIONS),
        ("PR draft", PR_DRAFT, REQUIRED_PR_SECTIONS),
    ]:
        try:
            content = _read(path)
        except FileNotFoundError as exc:
            print(f"[FAIL] {exc}")
            failures.append(str(exc))
            continue
        failures.extend(_check_sections(label, content, sections))
        failures.extend(_check_references(label, content))

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).")
        return 1

    print("\n[OK] All GitHub issue/PR drafts are structurally valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
