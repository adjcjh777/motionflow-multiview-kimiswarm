#!/usr/bin/env bash
# v26 temporal geometry fusion local smoke test.
# Runs a quick syntax check on the new module and then executes the unit tests.
set -euo pipefail

PYTHON=${PYTHON:-python}

echo "==> Syntax check motionflow_mv/fusion/temporal_geometry_fusion_v26.py"
$PYTHON -m py_compile motionflow_mv/fusion/temporal_geometry_fusion_v26.py

echo "==> Running v26 unit tests"
$PYTHON -m pytest tests/test_temporal_geometry_fusion_v26.py -q

echo "==> v26 smoke test passed"
