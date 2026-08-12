# Cleanup Plan for `tmp/` and Old Smoke Outputs

> **Status:** Draft — requires human review before execution.  
> **Scope:** Local working tree only (`D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`).  
> **No A800-D or Docker operations.**

## 1. Why cleanup is needed

The repository has accumulated a large amount of temporary diagnostic data, smoke-test artifacts, and stale checkpoints in two locations:

- `tmp/` — scratch scripts, smoke checkpoints, repro-gate data, swarm iteration logs, hyper-parameter search scratch.
- `outputs/` — training outputs, many of which are old smoke-run artifacts from model variants (`v25`–`v79`) that used the circular H36M labels and are no longer scientifically meaningful.

Cleaning these up will:

1. Reclaim disk space (currently `tmp/` ~ several GB, `outputs/` ~ many GB).
2. Remove artifacts trained on the circular-label H36M data, avoiding accidental reuse.
3. Make the remaining `outputs/` directory reflect only the **true-GT** and **Shelf/Campus** baselines relevant to the CVPR 2027 story.

## 2. What must NOT be touched

| Category | Examples | Reason |
|----------|----------|--------|
| True-GT H36M data | `data/h36m_true_gt/` | True labels; required for current baselines. |
| Valid Shelf/Campus data | `data/webbridge/shelf_campus_detected/` | Non-circular, used for true-GT evaluation. |
| Ongoing training outputs | Anything matching current `agent-51` / `agent-67` runs (see TaskList) | Work in flight. |
| Active config / source files | `configs/`, `motionflow_mv/`, `scripts/`, `tests/`, `docs/` | Out of scope. |
| A800-D / Docker paths | `/mnt/nvme0n1/zhangzy/projects`, Docker `motionflow` service | Read-only per project policy. |

## 3. Inventory findings (2026-08-11)

### 3.1 `tmp/` directory

- ~270+ top-level files and directories.
- Contains many one-off diagnostic `.py` scripts (`check_*.py`, `inspect_*.py`, `convert_*.py`).
- Contains stale smoke checkpoints: `smoke_v25*.pth`, `v25_*.pth`, `v26_*.pth`, `bayesian_tri_v*_smoke.pth`, etc.
- Contains old repro-gate and hyper-parameter search scratch: `hp_search_smoke/`, `hp_search_real/`, `reprojgate_smoke_data/`.
- Contains swarm iteration logs: `swarm_iter_next/`, `swarm_iter23_logs/`.
- Contains old DLT baselines for Shelf/Campus: `campus_seq1_dlt_baseline.npz`, `shelf_seq1_dlt_baseline.npz`.

### 3.2 `outputs/` directory

- Thousands of files, many GB total.
- Dominated by smoke-run `.pth` files from variants `v25`–`v79`.
- Includes `_smoke*.pth` / `_smoke_final.pth` variants.
- Includes directories `benchmark_v25_smoke/`, `benchmark_v25_smoke_dry/`, `benchmark_v25_smoke_v2/`.
- Retains files from the circular-label era that should not be used for model selection.

## 4. Cleanup strategy

Use a **three-pass** approach:

1. **Audit & protect** — generate a manifest and mark protected files.
2. **Archive** — move candidate deletions to a dated archive directory.
3. **Delete** — remove the archive after a validation period.

### 4.1 Pass 1: Audit & protect

Run the following dry-run commands and save manifests:

```bash
# Manifests
tmp_manifest="tmp/manifest_$(date +%Y%m%d).txt"
outputs_manifest="outputs/manifest_$(date +%Y%m%d).txt"

find tmp/ -type f -printf '%s %p\n' | sort -n > "$tmp_manifest"
find outputs/ -type f -printf '%s %p\n' | sort -n > "$outputs_manifest"

# Identify candidate smoke/old artifacts
find outputs/ -type f \( -name '*_smoke*.pth' -o -name '*_smoke*.log' -o -name '*_smoke*.json' \) > outputs/smoke_candidates_$(date +%Y%m%d).txt
find tmp/ -type f \( -name '*_smoke*.pth' -o -name '*_smoke*.npz' -o -name 'check_*.py' -o -name 'inspect_*.py' \) > tmp/smoke_candidates_$(date +%Y%m%d).txt
```

Before deleting anything, **protect**:

- Any file currently being written to (check with `lsof` / `fuser` if available on WSL).
- Any file referenced by an active training run (`agent-51`, `agent-67`).
- Any `.pth` or `.npz` explicitly listed in `docs/results_true_gt_*.md` as a baseline artifact.

### 4.2 Pass 2: Archive

Move candidates to a dated archive directory rather than deleting immediately:

```bash
archive_root="archive_cleanup_$(date +%Y%m%d)"
mkdir -p "$archive_root/tmp" "$archive_root/outputs"

# Example for tmp/ smoke checkpoints (adjust after reviewing manifest)
# xargs -I{} mv {} "$archive_root/tmp/{}" < tmp/smoke_candidates_*.txt

# Example for outputs/ smoke artifacts
# xargs -I{} mv {} "$archive_root/outputs/{}" < outputs/smoke_candidates_*.txt
```

Archive the candidate manifests in the same place for traceability.

### 4.3 Pass 3: Delete

After a validation period (recommend **7 days**), permanently delete the archive if:

- No active code references any path inside it.
- No training run has failed due to a missing file.
- The team has confirmed the remaining `outputs/` still contains the baselines they need.

```bash
# After validation period
rm -rf "$archive_root"
```

## 5. Specific cleanup rules

### 5.1 `tmp/` rules

| Keep | Delete / Archive |
|------|------------------|
| `tmp/t14_test/` if still used by CI | Empty / one-off diagnostic `.py` scripts (`check_*.py`, `inspect_*.py`, `convert_*.py`) |
| Recent true-GT validation `.npz` (e.g. `campus_seq1_dlt_baseline.npz` if referenced) | Old smoke checkpoints: `smoke_v25*.pth`, `v25_*.pth`, `v26_*.pth`, `bayesian_tri_v*_smoke.pth` |
| `tmp/swarm_iter_next/` if still active | `tmp/swarm_iter23_logs/` (older swarm runs) |
| `tmp/reprojgate_smoke_data/` if still needed for repro gate | `tmp/hp_search_smoke/`, `tmp/hp_search_v2_smoke*` if no longer needed |
| Logs tied to active runs | Zero-byte log files (`critic_diag.log`, `critic_v1.log`, etc.) |

### 5.2 `outputs/` rules

| Keep | Delete / Archive |
|------|------------------|
| Files explicitly in `docs/results_true_gt_h36m.md` or `docs/results_true_gt_shelf_campus.md` | All `*_smoke*.pth` / `*_smoke*.log` / `*_smoke*.json` not listed as protected |
| `omniview_fusion_v25_h36m_true_gt_medium*.pth` and related true-GT runs | Pre-true-GT H36M smoke runs (circular-label era) |
| `a800_h36m_reg/` and latest `a800_*.pth` if they belong to active runs | `benchmark_v25_smoke/`, `benchmark_v25_smoke_dry/`, `benchmark_v25_smoke_v2/` unless actively referenced |
| Latest stable checkpoints needed for paper figures | Duplicate `_final.pth` if the non-final pair is also present and unneeded |
| DLT / Iskakov baseline results on true GT | Anything with a duplicate older timestamp that is not the latest run |

## 6. Validation checklist

Before and after the cleanup, run:

- [ ] `du -sh tmp/ outputs/` — record space before/after.
- [ ] `find tmp/ outputs/ -type f | wc -l` — record file counts.
- [ ] Verify active background tasks still run (`TaskList`, `tmux ls`, `ps`).
- [ ] Re-run the smoke test suite or a minimal data diagnostic to ensure nothing critical was removed.
- [ ] Confirm `git status` is clean (only deletions, no source changes).

## 7. Rollback plan

- Keep the dated archive for 7 days.
- If a file is found to be missing, restore from the archive before it is deleted.
- Document any restored files in this plan or in a follow-up note.

## 8. Proposed execution commands (review before running)

```bash
#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +%Y%m%d)
ARCHIVE="archive_cleanup_${DATE}"

echo "=== Generating manifests ==="
find tmp/ -type f -printf '%s %p\n' | sort -n > "tmp/manifest_${DATE}.txt"
find outputs/ -type f -printf '%s %p\n' | sort -n > "outputs/manifest_${DATE}.txt"

echo "=== Identifying smoke candidates ==="
find outputs/ -type f \( -name '*_smoke*.pth' -o -name '*_smoke*.log' -o -name '*_smoke*.json' \) > "outputs/smoke_candidates_${DATE}.txt"
find tmp/ -type f \( -name '*_smoke*.pth' -o -name '*_smoke*.npz' -o -name 'check_*.py' -o -name 'inspect_*.py' \) > "tmp/smoke_candidates_${DATE}.txt"

echo "=== Creating archive ==="
mkdir -p "${ARCHIVE}/tmp" "${ARCHIVE}/outputs"
# After manual review, uncomment the xargs mv lines below:
# xargs -I{} mv "{}" "${ARCHIVE}/tmp/{}" < "tmp/smoke_candidates_${DATE}.txt"
# xargs -I{} mv "{}" "${ARCHIVE}/outputs/{}" < "outputs/smoke_candidates_${DATE}.txt"

echo "=== Cleanup staged in ${ARCHIVE}. Review before final deletion. ==="
```

## 9. Open questions / blockers

1. Which smoke-run `.pth` files (if any) are still needed as paper-story baselines? Need confirmation from the paper lead before archiving `outputs/*_smoke*.pth`.
2. Are `tmp/swarm_iter_next/` and `tmp/t14_test/` still actively used by CI or the orchestrator? If yes, they should be excluded from cleanup.
3. Do any scripts under `scripts/` hard-code paths to `tmp/` artifacts that would break after cleanup? A grep for `tmp/` in `scripts/` and `motionflow_mv/` is recommended before execution.
4. Is there enough disk space to even stage an archive, or should we delete directly? If space is tight, consider a shorter validation window or per-file deletion with `git`-like manifest.
