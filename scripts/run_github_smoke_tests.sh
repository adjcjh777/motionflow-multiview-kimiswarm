#!/usr/bin/env bash
# Local smoke-test runner for the GitHub Actions smoke-test workflow.
#
# Mirrors the commands in .github/workflows/smoke_tests.yml so that the same
# test subset can be validated quickly on the local RTX 4090 (or any WSL/CPU
# environment) before pushing.
#
# Usage:
#     bash scripts/run_github_smoke_tests.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"

echo "==> Running GitHub smoke tests from $ROOT"

echo "==> pytest tests/test_multiview_geometry_fusion_v25.py -q"
pytest "$ROOT/tests/test_multiview_geometry_fusion_v25.py" -q

echo "==> pytest tests/test_auto_eval_when_ready.py -q"
pytest "$ROOT/tests/test_auto_eval_when_ready.py" -q

echo "==> All GitHub smoke tests passed."
