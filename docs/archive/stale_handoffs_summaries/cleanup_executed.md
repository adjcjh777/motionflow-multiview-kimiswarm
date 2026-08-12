# Cleanup Executed: tmp/ and Old Smoke Outputs

> **Date:** 2026-08-11
> **Executed by:** agent cleanup subagent
> **Scope:** Local working tree (`D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`)
> **Policy:** Archive only — no permanent deletions. A800-D and Docker paths were not touched.

## Summary

Executed a conservative, archive-only cleanup of stale smoke-run artifacts in `tmp/` and `outputs/`, per `docs/cleanup_plan.md`.

| Metric | Before | After | Archived |
|---|---:|---:|---:|
| `tmp/` | 1.9 G | 1.9 G | 48 small files |
| `outputs/` | 3.6 G | 1.9 G | 528 files + 3 dirs (~1.7 G) |
| Total files (tmp + outputs) | 2068 | 1483 | 576 files + 3 dirs* |

\* The raw file-count drop is 585, but the active v25 training run created ~9 new files in `outputs/` during the ~4-minute cleanup window. The archive itself contains 576 files plus 3 directories.

All files were **moved** to `archive_cleanup_20260811/` rather than deleted, preserving the original relative paths. The archive can be reviewed and, if needed, restored before any permanent deletion step.

## What was archived

### `tmp/` (48 files)

- Smoke checkpoints: `*_smoke*.pth`, `*_smoke*.npz`
- One-off diagnostic scripts: `check_*.py`, `inspect_*.py`
- Examples:
  - `tmp/bayesian_tri_v2_stabilized_smoke.pth`
  - `tmp/check_*.py`
  - `tmp/inspect_*.py`
  - `tmp/mpi_s01_seq01_smoke.npz`

### `outputs/` (528 files + 3 directories)

- All `*_smoke*.pth`, `*_smoke*.log`, and `*_smoke*.json` not explicitly protected.
- Old circular-label smoke directories:
  - `outputs/benchmark_v25_smoke`
  - `outputs/benchmark_v25_smoke_dry`
  - `outputs/benchmark_v25_smoke_v2`

## Protected / preserved items

The following were explicitly excluded from the cleanup:

| Category | Items |
|---|---|
| Active training run (agent-51) | `outputs/omniview_fusion_v25_h36m_true_gt_medium.{pth,log}` |
| Active tail process | `outputs/v39_rcgr_smoke_local_4090.log` |
| True-GT H36M baselines | `outputs/iskakov_h36m_true_gt.{pth,log}`; `outputs/omniview_fusion_v80_h36m_true_gt_smoke.{pth,log,config.json}` and `_final.pth` |
| Shelf/Campus detected baselines | `outputs/iskakov_shelf_campus_detected.*`; `outputs/omniview_fusion_v{25,57,80}_shelf_campus_detected_smoke.*` and `_final.pth` |
| Shelf/Campus long run | `outputs/omniview_fusion_v80_shelf_campus_detected_long.*` |
| Active tmp directories | `tmp/swarm_iter_next/`; `tmp/reprojgate_smoke.pth` |
| A800 outputs | `outputs/a800_h36m_reg/` |

## Archive location

```
archive_cleanup_20260811/
├── cleanup_summary_20260811.txt         # execution log with before/after metrics
├── tmp_manifest_20260811.txt            # full pre-cleanup manifest of tmp/
├── outputs_manifest_20260811.txt        # full pre-cleanup manifest of outputs/
├── tmp_smoke_candidates_*.txt           # candidate lists before/after filtering
├── outputs_smoke_candidates_*.txt
├── tmp/                                 # archived tmp files (preserved paths)
└── outputs/                             # archived outputs files (preserved paths)
```

Total archive size: **~1.7 GB**.

## Validation checks performed

- [x] Active GPU training run (agent-51, v25 medium) is still running and logging.
- [x] Protected baseline artifacts are still present in `outputs/`.
- [x] `git status` shows only the expected new archive/script files and existing project changes; no source files were modified by the cleanup.
- [x] No files currently held open by active processes were moved.

## How to restore

If any archived file is needed, restore it from the archive. For example:

```bash
# Restore a specific file
mv archive_cleanup_20260811/outputs/some_smoke_run.pth outputs/some_smoke_run.pth

# Restore an entire directory
mv archive_cleanup_20260811/outputs/benchmark_v25_smoke outputs/benchmark_v25_smoke
```

Document any restoration in this file or in a follow-up note.

## Recommended next step

Per `docs/cleanup_plan.md` section 4.3, keep the archive for a validation period (suggested **7 days**). After confirming no active code or run depends on the archived files, the archive can be permanently deleted with:

```bash
rm -rf archive_cleanup_20260811
rm cleanup_safe_20260811.sh
```

## Blockers / caveats

- `tmp/` remained ~1.9 GB because the archived tmp files were small (diagnostic scripts and small smoke checkpoints). Most of `tmp/` consists of larger scratch data not covered by the smoke-artifact rules (e.g., feature caches, HP-search scratch, non-smoke `.npz` files). A follow-up cleanup focused on those categories would be needed to reduce `tmp/` further.
- The cleanup did not touch `outputs/a800_h36m_reg/` or any A800-D/Docker paths, per project policy.
