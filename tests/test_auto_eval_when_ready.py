"""CPU smoke tests for scripts/auto_eval_when_ready.py."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def script_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "scripts" / "auto_eval_when_ready.py"


@pytest.fixture
def project_tmp(script_path: Path) -> Path:
    root = script_path.parents[1]
    d = root / "tmp" / "pytest_auto_eval"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _run(script: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    root = script.parents[1]
    cmd = [sys.executable, script.relative_to(root).as_posix(), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=root.as_posix(),
        check=check,
    )


def test_dry_run_detects_completed_baseline(script_path: Path, project_tmp: Path) -> None:
    log = project_tmp / "omniview_fusion_v99_fake.log"
    # The script maps the experiment stem to a fullscale launch script.
    fullscale = script_path.parent / "run_omniview_fusion_v99_fake_fullscale.sh"
    fullscale.write_text("#!/bin/bash\necho fullscale\n")
    fullscale.chmod(0o755)

    try:
        log.write_text(
            "Epoch 1: val_MPJPE=123.45mm\n"
            "Epoch 2: val_MPJPE=110.00mm\n"
        )

        result = _run(
            script_path,
            "--log-dir", str(project_tmp),
            "--log-glob", "*.log",
            "--once",
            "--dry-run",
            "--eval-command", "echo eval-test",
        )
        assert result.returncode == 0, result.stderr
        text = result.stdout + result.stderr
        assert "Detected completed baseline" in text
        assert "DRY-RUN: would run eval" in text
        assert "DRY-RUN: would launch" in text
        assert "omniview_fusion_v99_fake" in text
    finally:
        fullscale.unlink(missing_ok=True)


def test_dry_run_does_not_create_done_marker(script_path: Path, project_tmp: Path) -> None:
    log = project_tmp / "omniview_fusion_v95_fake.log"
    log.write_text("Epoch 1: val_MPJPE=77.00mm\n")
    marker = Path(str(log) + ".auto_eval_done")

    result = _run(script_path, "--log-dir", str(project_tmp), "--log-glob", "*.log", "--once", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "Dry-run should not create an auto_eval_done marker"


def test_does_not_retrigger_already_processed_log(script_path: Path, project_tmp: Path) -> None:
    log = project_tmp / "omniview_fusion_v99_fake.log"
    log.write_text("Epoch 1: val_MPJPE=123.45mm\n")
    marker = Path(str(log) + ".auto_eval_done")
    time.sleep(0.05)
    marker.touch()

    result = _run(script_path, "--log-dir", str(project_tmp), "--log-glob", "*.log", "--once", "--dry-run")
    assert result.returncode == 0, result.stderr
    text = result.stdout + result.stderr
    assert "Detected completed baseline" not in text
    assert "DRY-RUN: would launch" not in text


def test_skips_log_without_val_mpjpe(script_path: Path, project_tmp: Path) -> None:
    log = project_tmp / "omniview_fusion_v98_no_val.log"
    log.write_text("Epoch 1: train_loss=0.5\n")

    result = _run(script_path, "--log-dir", str(project_tmp), "--log-glob", "*.log", "--once", "--dry-run")
    assert result.returncode == 0, result.stderr
    text = result.stdout + result.stderr
    assert "Detected completed baseline" not in text
    assert "v98_no_val" not in text


def test_json_summary_written(script_path: Path, project_tmp: Path) -> None:
    log = project_tmp / "omniview_fusion_v97_fake.log"
    log.write_text("Epoch 1: val_MPJPE=99.00mm\n")
    summary_path = project_tmp / "summary.json"

    result = _run(
        script_path,
        "--log-dir", str(project_tmp),
        "--log-glob", "*.log",
        "--once",
        "--dry-run",
        "--json-summary", str(summary_path),
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text())
    assert summary["scanned"] >= 1
    assert "omniview_fusion_v97_fake" in summary["processed"]


def test_second_instance_exits_due_to_lock(script_path: Path, project_tmp: Path) -> None:
    log = project_tmp / "omniview_fusion_v96_fake.log"
    log.write_text("Epoch 1: val_MPJPE=88.00mm\n")

    # Use a unique lock file to avoid conflicts with other tests/processes.
    lock = project_tmp / "motionflow_auto_eval_py.lock"
    # Write the current test process pid so the lock appears held by a live process.
    lock.write_text(str(os.getpid()))

    result = _run(
        script_path,
        "--log-dir", str(project_tmp),
        "--log-glob", "*.log",
        "--once",
        "--dry-run",
        "--lock-file", str(lock),
    )
    # Should exit gracefully because another instance is "running".
    assert result.returncode == 0, result.stderr
    assert "already running" in (result.stdout + result.stderr).lower()
