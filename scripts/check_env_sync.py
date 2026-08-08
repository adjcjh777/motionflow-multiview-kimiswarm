#!/usr/bin/env python3
"""A800-D vs local environment and dependency sync checker.

This script performs a read-only comparison between the local WSL/RTX 4090
environment and the A800-D environment used for full-scale training runs.  It
parses ``requirements.txt``, compares installed package versions, checks that
the v25 geometry-fusion module imports cleanly on both sides, and writes a
Markdown report.

Usage
-----
    # Local-only check (useful when A800 is unreachable or you just want a quick
    # sanity check on the 4090 box before pushing code):
    python scripts/check_env_sync.py --local-only

    # Full A800 vs local comparison:
    python scripts/check_env_sync.py

The report is written to ``docs/swarm_iter_next/environment_sync_report.md``
by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project-specific defaults
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
REPORT_DIR = REPO_ROOT / "docs" / "swarm_iter_next"
DEFAULT_REPORT_PATH = REPORT_DIR / "environment_sync_report.md"

SSH_HOST = "a800-D"
REMOTE_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
REMOTE_VENV_PYTHON = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python"

# Packages that are critical for the v25 iteration and should appear in both
# environments with matching versions.
CRITICAL_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pyyaml",
    "pytest",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Package:
    name: str
    version: str


@dataclass
class Environment:
    label: str
    python_version: str = "unknown"
    torch_version: str = "unknown"
    cuda_version: Optional[str] = None
    gpu_info: List[str] = field(default_factory=list)
    packages: Dict[str, Package] = field(default_factory=dict)
    v25_import_ok: bool = False
    v25_import_msg: str = "not checked"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: List[str], *, timeout: int = 60, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError as exc:
        return -1, "", f"{exc}"
    return result.returncode, result.stdout, result.stderr


def ssh_cmd(remote_cmd: str, host: str, timeout: int = 60) -> Tuple[int, str, str]:
    """Execute a command on the remote host via SSH."""
    return run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, remote_cmd],
        timeout=timeout,
    )


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def parse_requirements(path: Path) -> List[Tuple[str, str, str]]:
    """Parse ``requirements.txt`` into a list of (raw_name, op, version).

    Lines such as ``numpy>=1.24.0`` become ``("numpy", ">=", "1.24.0")``.
    Comments, blank lines and ``--extra-index-url`` lines are ignored.
    """
    specs: List[Tuple[str, str, str]] = []
    if not path.exists():
        return specs
    pattern = re.compile(r"^([A-Za-z0-9_.\-]+)\s*([<>=!~]+)\s*(.+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = pattern.match(line)
        if match:
            specs.append((match.group(1), match.group(2), match.group(3).strip()))
    return specs


def _version_key(version: str) -> Tuple[int, ...]:
    """Turn a version string into a comparable tuple of integers.

    Build/local segments (after ``+``) and pre/post-release segments are
    stripped/ignored for simplicity, which is sufficient for the small set of
    pinned packages in this project.
    """
    clean = version.split("+")[0].split("-")[0]
    parts: List[int] = []
    for part in clean.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _satisfies(installed: str, op: str, spec: str) -> bool:
    """Minimal PEP-440-ish spec evaluation for the project's requirements."""
    if op == "==":
        return _version_key(installed) == _version_key(spec)
    if op == ">=":
        return _version_key(installed) >= _version_key(spec)
    if op == "<=":
        return _version_key(installed) <= _version_key(spec)
    if op == ">":
        return _version_key(installed) > _version_key(spec)
    if op == "<":
        return _version_key(installed) < _version_key(spec)
    if op == "!=":
        return _version_key(installed) != _version_key(spec)
    if op == "~=":
        # Compatible release: same major/minor, installed >= spec.
        return _version_key(installed)[:2] == _version_key(spec)[:2] and _version_key(installed) >= _version_key(spec)
    # Unknown operator: assume satisfied, but the package is present.
    return True


def collect_pip_packages(python: str, host_label: str) -> Dict[str, Package]:
    """Collect installed packages using the given Python interpreter."""
    packages: Dict[str, Package] = {}
    ret, out, err = run([python, "-m", "pip", "list", "--format=json"], timeout=120)
    if ret != 0:
        print(f"[warn] Could not query packages for {host_label}: {err}", file=sys.stderr)
        return packages
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        print(f"[warn] Could not parse package list for {host_label}", file=sys.stderr)
        return packages
    for item in data:
        name = item["name"]
        version = item["version"]
        packages[normalize_name(name)] = Package(name=name, version=version)
    return packages


def collect_python_version(python: str) -> str:
    ret, out, _ = run([python, "--version"], timeout=30)
    return out.strip() if ret == 0 else "unknown"


def collect_torch_version(python: str) -> str:
    ret, out, _ = run(
        [python, "-c", "import torch; print(torch.__version__)"],
        timeout=30,
    )
    return out.strip() if ret == 0 else "unknown"


def collect_cuda_version(python: str) -> str:
    ret, out, _ = run(
        [python, "-c", "import torch; print(torch.version.cuda or 'N/A')"],
        timeout=30,
    )
    return out.strip() if ret == 0 else "unknown"


def collect_gpu_info() -> List[str]:
    ret, out, _ = run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        timeout=30,
    )
    if ret != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def check_v25_import(python: str, repo_root: Path, label: str) -> Tuple[bool, str]:
    """Try importing the v25 geometry-fusion module."""
    cmd = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "import motionflow_mv.fusion.multiview_geometry_fusion_v25 as m; "
        "print(m.__file__)"
    )
    ret, out, err = run([python, "-c", cmd], timeout=60)
    if ret == 0:
        return True, f"OK ({out.strip()})"
    return False, err.strip() or "import failed"


def collect_local_environment() -> Environment:
    env = Environment(label="local")
    python = sys.executable
    env.python_version = collect_python_version(python)
    env.packages = collect_pip_packages(python, "local")
    env.torch_version = collect_torch_version(python)
    env.cuda_version = collect_cuda_version(python)
    env.gpu_info = collect_gpu_info()
    env.v25_import_ok, env.v25_import_msg = check_v25_import(python, REPO_ROOT, "local")
    return env


def collect_remote_environment(host: str, repo: str, venv_python: str) -> Environment:
    env = Environment(label=f"a800 ({host})")

    # Test SSH reachability first.
    ret, _, err = ssh_cmd("echo ping", host, timeout=10)
    if ret != 0:
        env.v25_import_msg = f"SSH unreachable: {err.strip()}"
        return env

    # We cannot easily use ``sys.executable`` on the remote, so we call the
    # known venv python directly.
    env.python_version = f"{venv_python} -> " + _remote_python_version(host, venv_python)
    env.packages = _remote_pip_packages(host, venv_python)
    env.torch_version = _remote_torch_version(host, venv_python)
    env.cuda_version = _remote_cuda_version(host, venv_python)
    env.gpu_info = _remote_gpu_info(host)
    env.v25_import_ok, env.v25_import_msg = _remote_v25_import(host, venv_python, repo)
    return env


def _remote_python_version(host: str, python: str) -> str:
    ret, out, _ = ssh_cmd(f"{python} --version", host, timeout=30)
    return out.strip() if ret == 0 else "unknown"


def _remote_pip_packages(host: str, python: str) -> Dict[str, Package]:
    ret, out, _ = ssh_cmd(f"{python} -m pip list --format=json", host, timeout=120)
    if ret != 0:
        return {}
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {}
    return {normalize_name(item["name"]): Package(name=item["name"], version=item["version"]) for item in data}


def _remote_torch_version(host: str, python: str) -> str:
    ret, out, _ = ssh_cmd(f"{python} -c 'import torch; print(torch.__version__)'", host, timeout=30)
    return out.strip() if ret == 0 else "unknown"


def _remote_cuda_version(host: str, python: str) -> str:
    ret, out, _ = ssh_cmd(f"{python} -c 'import torch; print(torch.version.cuda or \"N/A\")'", host, timeout=30)
    return out.strip() if ret == 0 else "unknown"


def _remote_gpu_info(host: str) -> List[str]:
    ret, out, _ = ssh_cmd(
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        host,
        timeout=30,
    )
    if ret != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _remote_v25_import(host: str, python: str, repo: str) -> Tuple[bool, str]:
    cmd = (
        f"cd {repo} && "
        f"{python} -c "
        f"\"import sys; sys.path.insert(0, '{repo}'); "
        f"import motionflow_mv.fusion.multiview_geometry_fusion_v25 as m; print(m.__file__)\""
    )
    ret, out, err = ssh_cmd(cmd, host, timeout=60)
    if ret == 0:
        return True, f"OK ({out.strip()})"
    return False, err.strip() or "import failed"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_gpu_info(info: List[str]) -> str:
    if not info:
        return "*No GPU info available*"
    return "\n".join(f"- `{line}`" for line in info)


def build_report(
    local: Environment,
    remote: Optional[Environment],
    requirements: List[Tuple[str, str, str]],
    report_path: Path,
) -> Tuple[str, int, int]:
    warnings = 0
    errors = 0
    lines: List[str] = [
        "# Environment Sync Report (A800 vs Local)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "**Script:** `scripts/check_env_sync.py`",
        "",
        "## 1. Python / CUDA / GPU",
        "",
        "### Local",
        "",
        f"- Python: `{local.python_version}`",
        f"- PyTorch: `{local.torch_version}`",
        f"- CUDA (torch): `{local.cuda_version}`",
        "- GPUs:",
        _fmt_gpu_info(local.gpu_info),
        "",
    ]

    if remote is not None:
        lines.extend([
            f"### Remote ({remote.label})",
            "",
            f"- Python: `{remote.python_version}`",
            f"- PyTorch: `{remote.torch_version}`",
            f"- CUDA (torch): `{remote.cuda_version}`",
            "- GPUs:",
            _fmt_gpu_info(remote.gpu_info),
            "",
        ])

    lines.extend([
        "## 2. requirements.txt Compliance",
        "",
    ])
    for raw_name, op, req_version in requirements:
        norm = normalize_name(raw_name)
        local_pkg = local.packages.get(norm)
        if local_pkg is None:
            lines.append(f"- `{raw_name}` {op}{req_version} - **MISSING locally**")
            errors += 1
        elif not _satisfies(local_pkg.version, op, req_version):
            lines.append(f"- `{raw_name}` {op}{req_version} - local `{local_pkg.version}` **does not satisfy**")
            errors += 1
        else:
            lines.append(f"- `{raw_name}` {op}{req_version} - local `{local_pkg.version}` OK")
        if remote is not None:
            remote_pkg = remote.packages.get(norm)
            if remote_pkg is None:
                lines.append("  - **MISSING on remote**")
                errors += 1
            elif not _satisfies(remote_pkg.version, op, req_version):
                lines.append(f"  - remote `{remote_pkg.version}` **does not satisfy**")
                errors += 1

    lines.extend([
        "",
        "## 3. Critical Package Version Comparison",
        "",
        "| Package | Local | Remote | Match? |",
        "|---------|-------|--------|--------|",
    ])
    for name in CRITICAL_PACKAGES:
        local_ver = local.packages.get(name)
        remote_ver = remote.packages.get(name) if remote else None
        local_str = local_ver.version if local_ver else "missing"
        remote_str = remote_ver.version if remote_ver else "missing"
        match = "-" if remote is None else ("yes" if local_ver and remote_ver and local_ver.version == remote_ver.version else "**no**")
        lines.append(f"| `{name}` | `{local_str}` | `{remote_str}` | {match} |")
        if remote is not None and local_ver and remote_ver and local_ver.version != remote_ver.version:
            warnings += 1

    lines.extend([
        "",
        "## 4. v25 Module Import Check",
        "",
        f"- Local: {'OK' if local.v25_import_ok else 'FAIL'} - {local.v25_import_msg}",
    ])
    if not local.v25_import_ok:
        errors += 1
    if remote is not None:
        lines.append(f"- Remote: {'OK' if remote.v25_import_ok else 'FAIL'} - {remote.v25_import_msg}")
        if not remote.v25_import_ok:
            errors += 1

    lines.extend([
        "",
        "## 5. Recommendations",
        "",
    ])
    if errors:
        lines.append("- Resolve the errors above before launching a full A800 run.")
    if warnings:
        lines.append("- Review version mismatches; some may be intentional (e.g. CUDA builds).")
    if not errors and not warnings:
        lines.append("- Environments look in sync. Proceed with the v25 smoke test on the 4090.")
    lines.extend([
        "- Run the v25 small smoke config before scheduling the full A800 run.",
        "- Re-run this checker after any conda/pip update on either host.",
    ])

    report = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report, warnings, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare local and A800-D Python environments.")
    parser.add_argument("--local-only", action="store_true", help="Skip remote SSH checks.")
    parser.add_argument("--remote-host", default=SSH_HOST, help="A800 SSH host alias/address.")
    parser.add_argument("--remote-root", default=REMOTE_REPO, help="Remote repo root path.")
    parser.add_argument("--remote-venv", default=REMOTE_VENV_PYTHON, help="Remote python interpreter path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output Markdown report path.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Return non-zero on version mismatches as well.")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    requirements = parse_requirements(REQUIREMENTS_PATH)

    print("Collecting local environment ...", file=sys.stderr)
    local = collect_local_environment()

    remote: Optional[Environment] = None
    if not args.local_only:
        print(f"Collecting remote environment from {args.remote_host} ...", file=sys.stderr)
        remote = collect_remote_environment(args.remote_host, args.remote_root, args.remote_venv)
    else:
        print("Skipping remote checks (--local-only).", file=sys.stderr)

    print("Building report ...", file=sys.stderr)
    report, warnings, errors = build_report(local, remote, requirements, report_path)
    print(report)
    print(f"\nReport written to: {report_path}", file=sys.stderr)

    if errors:
        return 1
    if args.fail_on_warn and warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
