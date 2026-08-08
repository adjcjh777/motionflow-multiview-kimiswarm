#!/usr/bin/env python3
"""Validate v25 geometry-fusion GitHub templates.

Checks that the issue and PR templates exist, have valid YAML frontmatter,
and contain the sections/checklists required for the v25 round.

Usage:
    python scripts/validate_v25_templates.py

No training is performed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ISSUE_TEMPLATE = REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "v25_geometry_fusion_round.md"
PR_TEMPLATE = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"

REQUIRED_ISSUE_SECTIONS = [
    "## Experiment Summary",
    "## Model Configuration",
    "## Dataset & Resources",
    "## Metrics",
    "## Checklist",
    "## Related Issues / Runs",
    "## Notes / Observations",
]

REQUIRED_PR_SECTIONS = [
    "## v25 Geometry Fusion Round (if applicable)",
    "- [ ] `pytest tests/test_multiview_geometry_fusion_v25.py -q` passes",
    "- [ ] `v25_use_geometry_bundle_adjustment` starts as identity / no-op and is bounded",
]

REQUIRED_ISSUE_FRONTMATTER_KEYS = [
    "name:",
    "about:",
    "title:",
    "labels:",
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


def _check_frontmatter(path: Path, content: str) -> list[str]:
    """Check that the file has a valid YAML frontmatter block."""
    if not content.startswith("---"):
        print(f"[FAIL] {path.name} missing YAML frontmatter start")
        return ["missing frontmatter start"]

    # Extract first --- ... --- block.
    match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        print(f"[FAIL] {path.name} frontmatter block not terminated")
        return ["unterminated frontmatter"]

    failures: list[str] = []
    for key in REQUIRED_ISSUE_FRONTMATTER_KEYS:
        if key not in match.group(1):
            failures.append(f"missing frontmatter key: {key}")

    if failures:
        print(f"[FAIL] {path.name} frontmatter issues: {failures}")
    else:
        print(f"[OK] {path.name} frontmatter is valid")
    return failures


def main() -> int:
    failures: list[str] = []

    # Validate issue template.
    try:
        content = _read(ISSUE_TEMPLATE)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 1

    failures.extend(_check_frontmatter(ISSUE_TEMPLATE, content))
    failures.extend(_check_sections("Issue template", content, REQUIRED_ISSUE_SECTIONS))

    # Validate PR template.
    try:
        content = _read(PR_TEMPLATE)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 1

    failures.extend(_check_sections("PR template", content, REQUIRED_PR_SECTIONS))

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).")
        return 1

    print("\n[OK] v25 geometry-fusion templates are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
