#!/usr/bin/env python3
"""Read-only A800-D audit for the motionflow-multiview project.

Generates a Markdown inventory of available checkpoints, datasets, Docker
artifacts and run configs under:

    /mnt/nvme0n1/zhangzy/projects

The script only issues read-only SSH commands.  It never writes to the remote
host.

Usage
-----
    conda run -n mf python scripts/audit_a800_readonly_v5.py
    # or, without conda:
    python scripts/audit_a800_readonly_v5.py

The report is written to docs/swarm_iter5/a800_readonly_audit_report_v5.md.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

SSH_HOST = "a800-D"
REMOTE_ROOT = "/mnt/nvme0n1/zhangzy/projects"
REPORT_PATH = Path("docs/swarm_iter5/a800_readonly_audit_report_v5.md")


PROJECT_DIRS = [
    "motionflow-research-multiview-easymocap-robot-profiles",
    "motionflow-6df139c-build",
    "motionflow-f49d93e-build-KieqEr",
    "GVHMR",
    "GMR",
    "gmr-motionlab",
]


LOCAL_DATASETS = [
    "GVHMR/outputs/demo",
    "GMR/save",
    "motionflow-research-multiview-easymocap-robot-profiles/vendor/mjlab-elf3_beyongmimic/npz",
]


def ssh(cmd: str, timeout: int = 60) -> str:
    """Run a command on the A800-D host via SSH and return stdout."""
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST, cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        return f"ERROR (exit {result.returncode}): {result.stderr.strip()}"
    return result.stdout or ""


def fenced(title: str, body: str, lang: str = "text") -> str:
    return f"### {title}\n\n```{lang}\n{body.strip()}\n```\n"


def section(title: str) -> str:
    return f"\n## {title}\n"


def collect_directory_listing() -> str:
    out = ssh(f"ls -la {REMOTE_ROOT}")
    return fenced("Top-level directory listing", out)


def collect_sizes() -> str:
    out = ssh(f"du -sh {REMOTE_ROOT}/*")
    return fenced("Top-level directory sizes", out)


def collect_disk() -> str:
    out = ssh("df -h /mnt/nvme0n1 /mnt/nvme1n1p1 2>/dev/null || df -h /mnt/nvme0n1")
    return fenced("Disk usage for project/data volumes", out)


def collect_docker_ps() -> str:
    out = ssh(
        'docker ps -a --format "table {{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}"'
    )
    return fenced("Docker containers", out)


def collect_docker_images() -> str:
    out = ssh(
        'docker images --format "table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}" | head -40'
    )
    return fenced("Docker images (top 40)", out)


def collect_checkpoints() -> str:
    cmd = f"""
BASE={REMOTE_ROOT}
for d in {' '.join(PROJECT_DIRS)}; do
  echo "===== $d ====="
  find "$BASE/$d" -maxdepth 4 -type f \\( -iname "*.pth" -o -iname "*.pt" -o -iname "*.ckpt" \\) -printf '%p\\t%s\\n' 2>/dev/null | head -40
  echo
  echo "  sizes (MB):"
  find "$BASE/$d" -maxdepth 4 -type f \\( -iname "*.pth" -o -iname "*.pt" -o -iname "*.ckpt" \\) -printf '%s\\n' 2>/dev/null | awk '{{sum+=$1; count+=1}} END {{printf "  count=%d total_bytes=%d total_MB=%.2f\\n", count, sum, sum/1024/1024}}'
done
"""
    return fenced("Checkpoints / weights (.pth, .pt, .ckpt)", ssh(cmd, timeout=120))


def collect_datasets() -> str:
    out = ssh(
        f"""
BASE={REMOTE_ROOT}
for d in {' '.join(LOCAL_DATASETS)}; do
  echo "===== $d ====="
  find "$BASE/$d" -maxdepth 2 -type f \\( -iname "*.npz" -o -iname "*.pt" -o -iname "*.pkl" \\) -printf '%p\\t%s\\n' 2>/dev/null | head -30
done
""",
        timeout=120,
    )
    return fenced("Datasets / demo artifacts", out)


def collect_configs() -> str:
    out = ssh(
        f"""
BASE={REMOTE_ROOT}
for d in {' '.join(PROJECT_DIRS)}; do
  echo "===== $d ====="
  find "$BASE/$d" -maxdepth 3 -type f \\( -iname "*.yaml" -o -iname "*.yml" -o -iname "*.json" -o -iname "*.sh" -o -iname "Dockerfile*" \\) 2>/dev/null | head -25
  echo
done
""",
        timeout=120,
    )
    return fenced("Run configs / scripts / Dockerfiles", out)


def collect_container_config() -> str:
    out = ssh(
        """
docker inspect --format '{{json .Config}}' motionflow 2>/dev/null | \
python3 -c "import sys,json; c=json.load(sys.stdin); print(json.dumps({\\
'Entrypoint':c.get('Entrypoint'),'Cmd':c.get('Cmd'),'WorkingDir':c.get('WorkingDir'),\\
'Env':c.get('Env')}, indent=2, ensure_ascii=False))"
""",
        timeout=60,
    )
    return fenced("Running motionflow container config", out, lang="json")


def build_report() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts: List[str] = [
        "# A800-D Read-Only Audit Report (Swarm Iteration 5)\n",
        "**Date:**", f"{now}\n",
        "**Scope:**", f"`{REMOTE_ROOT}` on host `{SSH_HOST}`\n",
        "**Constraint:** Read-only operations only.\n",
        "**Audit script:**", "[`scripts/audit_a800_readonly_v5.py`](../scripts/audit_a800_readonly_v5.py)\n",
    ]

    parts.extend([
        section("1. Host & Storage"),
        collect_directory_listing(),
        collect_sizes(),
        collect_disk(),
    ])

    parts.extend([
        section("2. Docker Containers & Images"),
        collect_docker_ps(),
        collect_docker_images(),
        collect_container_config(),
    ])

    parts.extend([
        section("3. Checkpoints / Weights"),
        collect_checkpoints(),
    ])

    parts.extend([
        section("4. Datasets / Demo Artifacts"),
        collect_datasets(),
    ])

    parts.extend([
        section("5. Run Configs / Scripts / Dockerfiles"),
        collect_configs(),
    ])

    parts.extend([
        section("6. Implications for MotionFlow-MultiView"),
        """
- **GVHMR weights are present** (`gvhmr_siga24_release.ckpt`, `hmr2`,
  `vitpose`, `yolo`, `dpvo`, SMPL/SMPL-X body models). These can be used
  read-only for local per-view feature extraction and for adapting the IR
  pipeline.
- **GVHMR demo outputs** (`outputs/demo/*/hmr4d_results.pt`) are available for
  single-view IR adapter validation, but they are not multi-view calibrated
  data.
- **Multi-view / mocap NPZ files** live in `vendor/mjlab-elf3_beyongmimic/npz`
  and `GMR/save/`. They are robot retargeting artifacts, not standard 3D-HPE
  benchmarks, so they cannot directly train the MPI-INF-3DHP temporal fusion
  baseline without conversion/label verification.
- **Standard 3D-HPE datasets** (Human3.6M, MPI-INF-3DHP canonical .npz,
  Shelf/Campus, CMU Panoptic, 3DPW, AMASS) are **not present** in the audited
  tree. The existing local `data/webbridge/mpi_inf_3dhp/` remains the primary
  training source for the RayAttentionFusionModelTemporal smoke tests.
- **Docker images** (`elf3-trainer:*`, ~34 GB) target the ELF3 video-to-policy
  pipeline (CUDA 12.8 / PyTorch 2.9.1). They are not currently wired to the
  multiview fusion repo, but the Dockerfile documents the dependency baseline
  (PyTorch cu128, GVHMR, GMR, mjlab-elf3_beyongmimic, SMPL-X).
- **No new dependencies** are required to run this audit; the script only uses
  the Python standard library and an existing `ssh` binary.
""",
    ])

    return "\n".join(parts)


def main() -> None:
    print(f"Auditing {SSH_HOST}:{REMOTE_ROOT} ...")
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
