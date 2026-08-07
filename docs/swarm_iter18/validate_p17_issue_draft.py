#!/usr/bin/env python3
"""CPU smoke test for the P17 GitHub issue draft.

Usage:
    python docs/swarm_iter18/validate_p17_issue_draft.py

Checks:
    - The issue draft exists and is non-empty.
    - Required sections are present.
    - Referenced local files exist on disk (when already present).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUE_DRAFT = REPO_ROOT / "docs" / "swarm_iter18" / "P17_github_issue_draft.md"

REQUIRED_SECTIONS = [
    "## 1. Update on issue #70",
    "## 2. Current best result",
    "## 3. OmniMultiViewFusion",
    "### 3.2 Proposed architecture",
    "### 3.4 Training recipe",
    "### 3.5 Evaluation plan",
    "## 4. Deliverables for this issue",
    "## 5. Known blockers and risks",
    "## 6. Next steps",
    "## 7. Related files and references",
]

# Files that the draft claims should exist.  Some are intentionally future work
# and may not yet be on disk; only fail for files the prototype should already
# have created (i.e., the ones we create in this task).
EXPECTED_EXISTING_FILES = [
    "docs/swarm_iter18/P17_github_issue_draft.md",
    "docs/design_omniview_fusion.md",
    "docs/results_icra_cvpr_2027.md",
    "docs/icra_cvpr_2027_paper_story.md",
    "docs/swarm_iter_next/synthesis_2026_08_07.md",
    "docs/swarm_iter_next/20_agent_direction_review.md",
    "motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py",
    "motionflow_mv/fusion/principal_point_correction.py",
]


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _check_sections(content: str) -> list[str]:
    missing = [s for s in REQUIRED_SECTIONS if s not in content]
    if missing:
        print(f"[FAIL] Missing sections: {missing}")
    else:
        print("[OK] All required sections present.")
    return missing


def _check_references(content: str) -> list[str]:
    failures: list[str] = []
    for rel_path in EXPECTED_EXISTING_FILES:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            failures.append(f"{rel_path} (missing on disk)")
        elif rel_path not in content:
            failures.append(f"{rel_path} (not referenced)")
    if failures:
        print(f"[FAIL] Reference problems: {failures}")
    else:
        print("[OK] All expected files exist and are referenced.")
    return failures


def main() -> int:
    try:
        content = _read(ISSUE_DRAFT)
    except FileNotFoundError as exc:
        print(f"[FAIL] Could not read issue draft: {exc}")
        return 1

    if not content.strip():
        print("[FAIL] Issue draft is empty.")
        return 1

    print(f"[OK] Issue draft found: {ISSUE_DRAFT}")
    print(f"[OK] Draft length: {len(content)} chars / {len(content.splitlines())} lines")

    failures: list[str] = []
    failures.extend(_check_sections(content))
    failures.extend(_check_references(content))

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).")
        return 1

    print("\n[OK] P17 GitHub issue draft passes CPU smoke validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
