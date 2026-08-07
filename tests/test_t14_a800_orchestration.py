"""CPU smoke tests for T14 A800-D persistent training orchestration.

Verifies that ``scripts/run_omniview_fusion_v4_a800.sh`` and
``scripts/tmux_omniview_fusion_v4_a800.sh`` are executable, respond to
``--help`` / ``--smoke`` / ``--dry-run``, and use lock files correctly.
"""

import os
import shutil

BASH_BIN = shutil.which("bash") or "bash"
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = ROOT / "scripts" / "run_omniview_fusion_v4_a800.sh"
TMUX_SCRIPT = ROOT / "scripts" / "tmux_omniview_fusion_v4_a800.sh"


def _bash_path(path: Path) -> str:
    """Return a path that Git Bash / WSL bash can consume on Windows."""
    # Bash on Windows interprets backslashes as escape characters, so force
    # forward slashes.  Relative paths are safest and work on all platforms.
    try:
        rel = path.relative_to(ROOT)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _bash_abs(path: Path) -> str:
    """Return an absolute path with forward slashes for bash environment vars."""
    return str(path).replace("\\", "/")


@pytest.fixture(scope="module")
def temp_dirs():
    # Use a repo-relative scratch directory so Python and Git Bash agree on the
    # filesystem view.  We avoid tmp_path_factory because its Windows temp path
    # is not visible at the same absolute path from inside Git Bash.
    tmp = ROOT / "tmp" / "t14_test"
    (tmp / "lock").mkdir(parents=True, exist_ok=True)
    (tmp / "output").mkdir(parents=True, exist_ok=True)
    (tmp / "venv").mkdir(parents=True, exist_ok=True)
    return {
        "lock": tmp / "lock",
        "output": tmp / "output",
        "venv": tmp / "venv",
    }


def _make_executable_venv(venv_path: Path) -> None:
    """Create a tiny fake venv with an executable python shim.

    The shim is a bash script that mimics enough of ``python <script>
    --smoke`` for the orchestration tests.  We use a shell shim because on
    Windows the executable bit may not be honoured and we do not have a real
    Python binary at this path.
    """
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python_bin = bin_dir / "python"
    python_bin.write_text(
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$arg\" == '--smoke' ]]; then\n"
        "    echo 'trainer smoke ok'\n"
        "    exit 0\n"
        "  fi\n"
        "done\n"
        "exit 0\n"
    )
    python_bin.chmod(python_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestRunScript:
    """Tests for ``run_omniview_fusion_v4_a800.sh``."""

    def test_script_is_executable(self):
        assert RUN_SCRIPT.exists(), f"Missing {RUN_SCRIPT}"
        mode = RUN_SCRIPT.stat().st_mode
        # On Windows the executable bit may not be reflected by the filesystem,
        # but bash can still run the script explicitly.
        if sys.platform != "win32":
            assert mode & stat.S_IXUSR, f"{RUN_SCRIPT} is not executable"

    def test_help(self, temp_dirs):
        env = {
            "MF_ROOT": _bash_abs(ROOT),
            "MF_LOCK_DIR": _bash_abs(temp_dirs["lock"]),
            "MF_OUTPUT_DIR": _bash_abs(temp_dirs["output"]),
        }
        result = subprocess.run(
            [BASH_BIN, _bash_path(RUN_SCRIPT), "--help"],
            cwd=str(ROOT),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "Usage:" in result.stdout or "Persistent" in result.stdout

    def test_smoke_without_v4_trainer(self, temp_dirs):
        """Smoke must validate environment and report the missing v4 trainer."""
        _make_executable_venv(temp_dirs["venv"])
        env = {
            "MF_ROOT": _bash_abs(ROOT),
            "MF_VENV": _bash_abs(temp_dirs["venv"]),
            "MF_LOCK_DIR": _bash_abs(temp_dirs["lock"]),
            "MF_OUTPUT_DIR": _bash_abs(temp_dirs["output"]),
            "MF_ALLOWED_GPUS": "0",
            "MF_GPU": "0",
        }
        result = subprocess.run(
            [BASH_BIN, _bash_path(RUN_SCRIPT), "--smoke"],
            cwd=str(ROOT),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        # We expect 0 because the script should treat a missing trainer as a
        # warning during smoke mode.
        assert result.returncode == 0, f"--smoke failed: stdout={result.stdout}\nstderr={result.stderr}"
        combined = result.stdout + result.stderr
        assert "SMOKE / DRY-RUN mode" in combined or "Smoke" in combined

    def test_smoke_with_fake_trainer(self, temp_dirs):
        """When a trainer exists, smoke mode should invoke its ``--smoke``."""
        _make_executable_venv(temp_dirs["venv"])
        temp_dirs["output"].mkdir(parents=True, exist_ok=True)
        trainer = temp_dirs["output"] / "train_omniview_fusion_v4_webbridge_multi.py"
        trainer.write_text(
            "import sys\n"
            "if '--smoke' in sys.argv:\n"
            "    print('trainer smoke ok')\n"
            "    sys.exit(0)\n"
        )
        env = {
            "MF_ROOT": _bash_abs(ROOT),
            "MF_VENV": _bash_abs(temp_dirs["venv"]),
            "MF_LOCK_DIR": _bash_abs(temp_dirs["lock"]),
            "MF_OUTPUT_DIR": _bash_abs(temp_dirs["output"]),
            "MF_TRAINER": _bash_abs(trainer),
            "MF_ALLOWED_GPUS": "0",
            "MF_GPU": "0",
        }
        result = subprocess.run(
            [BASH_BIN, _bash_path(RUN_SCRIPT), "--smoke"],
            cwd=str(ROOT),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--smoke failed: stdout={result.stdout}\nstderr={result.stderr}"
        combined = result.stdout + result.stderr
        assert "trainer smoke ok" in combined or "Smoke run completed successfully" in combined

    def test_gpu_selection_respects_allowed_and_busy(self, temp_dirs):
        """GPU should be chosen from allowed list and skip busy GPUs."""
        _make_executable_venv(temp_dirs["venv"])
        env = {
            "MF_ROOT": _bash_abs(ROOT),
            "MF_VENV": _bash_abs(temp_dirs["venv"]),
            "MF_LOCK_DIR": _bash_abs(temp_dirs["lock"]),
            "MF_OUTPUT_DIR": _bash_abs(temp_dirs["output"]),
            "MF_ALLOWED_GPUS": "2,3",
            "MF_BUSY_GPUS": "2",
            "MF_GPU": "",
        }
        result = subprocess.run(
            [BASH_BIN, _bash_path(RUN_SCRIPT), "--smoke"],
            cwd=str(ROOT),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Selected GPU: 3" in (result.stdout + result.stderr)

    def test_unknown_argument_fails(self, temp_dirs):
        env = {
            "MF_ROOT": _bash_abs(ROOT),
            "MF_LOCK_DIR": _bash_abs(temp_dirs["lock"]),
            "MF_OUTPUT_DIR": _bash_abs(temp_dirs["output"]),
        }
        result = subprocess.run(
            [BASH_BIN, _bash_path(RUN_SCRIPT), "--bogus"],
            cwd=str(ROOT),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Unknown argument" in (result.stdout + result.stderr)


class TestTmuxScript:
    """Tests for ``tmux_omniview_fusion_v4_a800.sh``."""

    def test_script_is_executable(self):
        assert TMUX_SCRIPT.exists(), f"Missing {TMUX_SCRIPT}"
        mode = TMUX_SCRIPT.stat().st_mode
        if sys.platform != "win32":
            assert mode & stat.S_IXUSR, f"{TMUX_SCRIPT} is not executable"

    def test_dry_run(self, temp_dirs):
        env = {
            "MF_ROOT": _bash_abs(ROOT),
            "MF_LOCK_DIR": _bash_abs(temp_dirs["lock"]),
            "MF_OUTPUT_DIR": _bash_abs(temp_dirs["output"]),
        }
        result = subprocess.run(
            [BASH_BIN, _bash_path(TMUX_SCRIPT), "--dry-run"],
            cwd=str(ROOT),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--dry-run failed: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "[dry-run] Would start tmux session" in combined
        assert "run_omniview_fusion_v4_a800.sh" in combined

    def test_lock_prevents_duplicate_dry_run(self, temp_dirs):
        """A held lock should block a second dry-run launch.

        Skipped on Windows because Git Bash ``kill -0`` cannot signal a
        process started from the Windows Python runtime.
        """
        if sys.platform == "win32":
            pytest.skip("kill -0 cross-process check not reliable on Windows")
        env = {
            "MF_ROOT": _bash_abs(ROOT),
            "MF_LOCK_DIR": _bash_abs(temp_dirs["lock"]),
            "MF_OUTPUT_DIR": _bash_abs(temp_dirs["output"]),
        }
        temp_dirs["lock"].mkdir(parents=True, exist_ok=True)
        lock = temp_dirs["lock"] / "omniview_fusion_v4.tmux.lock"

        # Use a real background bash process so kill -0 can verify it is alive.
        sleeper = subprocess.Popen(
            [BASH_BIN, "-c", "sleep 30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            lock.write_text(str(sleeper.pid))

            result = subprocess.run(
                [BASH_BIN, _bash_path(TMUX_SCRIPT), "--dry-run"],
                cwd=str(ROOT),
                env={**os.environ, **env},
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "already running" in (result.stdout + result.stderr)
        finally:
            sleeper.kill()
            sleeper.wait()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
