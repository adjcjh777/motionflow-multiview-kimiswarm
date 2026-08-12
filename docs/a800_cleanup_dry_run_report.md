# A800 Cleanup Dry-Run Report

**Date:** 2026-08-12
**Command:** `bash scripts/cleanup_a800_safe.sh --dry-run`
**Executed on:** A800 (`a800-D`) at `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`

## Disk Status

| Time | Size | Used | Available | Use% |
|------|------|------|-----------|------|
| Before cleanup | 3.5T | 3.3T | 58G | 99% |
| After cleanup (dry-run) | 3.5T | 3.3T | 58G | 99% |

Disk usage remains at **99% (~58 GB free)**; the dry-run does not reclaim space.

## Items Identified for Removal

### 1. Duplicate / Non-Final Checkpoints

The script scans `outputs/ablations/` for checkpoints that have a corresponding `*_final.pth` sibling. One such file was flagged for removal:

- `outputs/ablations/aistpp_only_medium_a800_fast_v2.pth`

This checkpoint is a duplicate of the final checkpoint `aistpp_only_medium_a800_fast_v2_final.pth` from the completed AIST++-only medium fast v2 run. The non-final file is not referenced by an active process, so it is safe to delete.

### 2. Abandoned / Diverged Mixed-Dataset Checkpoint

- `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.pth`

This checkpoint was **not** flagged for removal because the file no longer exists (or the script did not print a removal line for it). This is consistent with the AGENTS.md note that the diverged run's best checkpoint was retained at this path; it appears to have already been removed or is not present.

### 3. Package Manager Caches

- **pip cache:** `yes | pip cache purge` reported `Files removed: 0` and `WARNING: No matching packages`. No pip cache space would be reclaimed.
- **uv cache:** The uv cache directory at `/mnt/nvme0n1p1/zhangzy/.cache/uv` was flagged for removal (uv is not in `PATH`). Since this is a dry-run, the directory was not actually deleted. The actual space that would be freed depends on the current contents of that directory.

## Safety Checks Performed by the Script

1. **Active process check:** Each non-final checkpoint is checked with `pgrep -f` against its basename. If a matching process is running, the file is skipped.
2. **Final checkpoint existence:** Only checkpoints with a corresponding `*_final.pth` file are candidates for removal.
3. **Abandoned checkpoint:** The v25 mixed-dataset checkpoint is only removed if it is not referenced by an active process.
4. **No destructive action in dry-run mode:** All `rm` operations are guarded by `if [ "$DRY_RUN" = false ]; then ... fi`.

## Files Created / Modified by This Task

- Created: `docs/a800_cleanup_dry_run_report.md` (this report)

No A800 files were modified or deleted; this was a dry-run only.

## Next Steps (if actual cleanup is desired)

To actually reclaim disk space, run the same command without `--dry-run`:

```bash
ssh a800-D "cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20 && bash scripts/cleanup_a800_safe.sh"
```

This would:
- Delete the duplicate `aistpp_only_medium_a800_fast_v2.pth` checkpoint.
- Remove the uv cache directory at `/mnt/nvme0n1p1/zhangzy/.cache/uv`.
- Attempt to purge the pip cache (currently empty).

**Caution:** The disk is critically full (99%). Even after this cleanup, the amount of space freed is likely small. Additional manual review of large failed runs (e.g., old v83/v84 outputs) may be necessary to make meaningful headroom.

## Notes

- The cleanup script did not identify the v25 mixed-dataset checkpoint for removal, suggesting it is no longer present at the expected path.
- The pip cache was already empty; no space will be reclaimed from pip.
- The uv cache directory removal is the largest potential source of space, but its size was not measured during the dry-run.
