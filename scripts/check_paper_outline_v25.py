#!/usr/bin/env python3
"""Validate the v25 paper outline markdown.

This is a minimal sanity check that the outline document:
- exists,
- contains the expected sections,
- has a parseable submission checklist.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REQUIRED_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Method",
    "Experiments and Results",
    "Discussion and Conclusion",
    "References",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    outline_path = repo_root / "docs" / "paper_corrected_outline_cvpr2027.md"

    if not outline_path.exists():
        print(f"ERROR: {outline_path} does not exist.", file=sys.stderr)
        return 1

    text = outline_path.read_text(encoding="utf-8")
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    if missing:
        print(f"ERROR: missing sections: {missing}", file=sys.stderr)
        return 1

    # Count checklist items and their states.
    checklist_items = re.findall(r"^\s*-\s+\[([ xX])\]", text, flags=re.MULTILINE)
    total = len(checklist_items)
    done = sum(1 for s in checklist_items if s.strip().lower() == "x")

    print(f"Validated {outline_path}")
    print(f"  Sections found: {len(REQUIRED_SECTIONS)}/{len(REQUIRED_SECTIONS)}")
    print(f"  Checklist items: {done}/{total} completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
