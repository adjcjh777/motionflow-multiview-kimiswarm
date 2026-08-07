"""CPU smoke tests for scripts/monitor_4090_v2_and_auto_eval.sh."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture
def script_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "scripts" / "monitor_4090_v2_and_auto_eval.sh"


@pytest.fixture
def project_tmp(script_path) -> Path:
    """Project-local temp directory so tests can pass bash-friendly relative paths."""
    root = script_path.parents[1]
    d = root / "tmp" / "pytest_monitor"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    # Best-effort cleanup; do not fail if files are locked.
    shutil.rmtree(d, ignore_errors=True)


LOCK_BASENAMES = (
    "motionflow_4090_monitor.lock",
    "motionflow_4090_auto_eval.lock",
    "motionflow_a800_eval.lock",
    "motionflow_auto_eval.lock",
)


def _run(
    script: Path,
    ckpt: Path,
    log: Path,
    lock_dir: Path | None = None,
    *,
    extra_args: list | None = None,
) -> subprocess.CompletedProcess:
    root = script.parents[1]
    script_rel = script.relative_to(root).as_posix()
    ckpt_rel = ckpt.relative_to(root).as_posix()
    log_rel = log.relative_to(root).as_posix()

    args = [
        "bash",
        script_rel,
        "--ckpt",
        ckpt_rel,
        "--log",
        log_rel,
    ]
    if lock_dir is not None:
        args += ["--lock-dir", lock_dir.relative_to(root).as_posix()]
    args += (extra_args or [])

    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=root.as_posix(),
    )


def test_dry_run_detects_new_checkpoint(script_path, project_tmp: Path):
    ckpt = project_tmp / "fake.pth"
    log = project_tmp / "monitor.log"
    ckpt.touch()

    result = _run(script_path, ckpt, log, lock_dir=project_tmp, extra_args=["--dry-run", "--once"])

    assert result.returncode == 0, result.stderr
    text = result.stdout + result.stderr
    assert "new checkpoint detected" in text
    assert "DRY-RUN: would launch 4090 eval" in text
    assert "monitor started" in text

    # Log file should also contain the dry-run marker.
    log_text = log.read_text()
    assert "DRY-RUN: would launch 4090 eval" in log_text


def test_up_to_date_checkpoint_skips_eval(script_path, project_tmp: Path):
    ckpt = project_tmp / "fake.pth"
    log = project_tmp / "monitor.log"
    ckpt.touch()
    done_file = Path(str(ckpt) + ".eval_done")
    # Create an eval_done marker newer than the checkpoint.
    time.sleep(0.1)
    done_file.touch()

    result = _run(script_path, ckpt, log, lock_dir=project_tmp, extra_args=["--dry-run", "--once"])

    assert result.returncode == 0, result.stderr
    text = result.stdout + result.stderr
    assert "checkpoint up-to-date" in text
    assert "DRY-RUN: would launch 4090 eval" not in text


def test_duplicate_watchdog_is_prevented(script_path, project_tmp: Path):
    ckpt = project_tmp / "fake.pth"
    log1 = project_tmp / "monitor1.log"
    log2 = project_tmp / "monitor2.log"
    ckpt.touch()

    root = script_path.parents[1]
    script_rel = script_path.relative_to(root).as_posix()

    # Start a long-running watchdog in the background.
    proc1 = subprocess.Popen(
        [
            "bash",
            script_rel,
            "--ckpt",
            ckpt.relative_to(root).as_posix(),
            "--log",
            log1.relative_to(root).as_posix(),
            "--lock-dir",
            project_tmp.relative_to(root).as_posix(),
            "--interval",
            "3600",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
        cwd=root.as_posix(),
    )
    try:
        # Give the first watchdog time to acquire its lock.
        time.sleep(0.5)
        result = _run(script_path, ckpt, log2, lock_dir=project_tmp)
        assert result.returncode == 0, result.stderr
        assert "already running" in (result.stdout + result.stderr).lower()
    finally:
        proc1.terminate()
        try:
            proc1.wait(timeout=2)
        except TimeoutError:
            proc1.kill()


def test_existing_eval_lock_defers_new_eval(script_path, project_tmp: Path):
    ckpt = project_tmp / "fake.pth"
    log = project_tmp / "monitor.log"
    ckpt.touch()

    # Create lock files in the same directory the script will use.
    for name in ("motionflow_4090_auto_eval.lock", "motionflow_a800_eval.lock", "motionflow_auto_eval.lock"):
        (project_tmp / name).touch()

    result = _run(script_path, ckpt, log, lock_dir=project_tmp, extra_args=["--dry-run", "--once"])

    assert result.returncode == 0, result.stderr
    text = result.stdout + result.stderr
    assert "new checkpoint detected" in text
    assert "eval lock present" in text
    assert "DRY-RUN: would launch 4090 eval" not in text


def test_help_flag(script_path):
    root = script_path.parents[1]
    script_rel = script_path.relative_to(root).as_posix()
    result = subprocess.run(
        ["bash", script_rel, "--help"],
        capture_output=True,
        text=True,
        cwd=root.as_posix(),
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
