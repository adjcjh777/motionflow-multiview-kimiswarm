# A800 Safe Cleanup Plan

> Generated from read-only inspection of `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`.
> No files were deleted or modified on A800 to produce this plan.

## Current disk status

```text
Filesystem       Size  Used Avail Use%
/dev/nvme0n1p1  3.5T  3.3T   58G  99%
```

## Active jobs that must NOT be disturbed

| PID       | GPU | Command / File in use                                      |
|-----------|-----|------------------------------------------------------------|
| 2218949   | 6   | `experiments/eval_variable_views.py --checkpoint outputs/ablations/v85_random_view_dropout_medium_a800.pth` |
| 2072252ff | —   | `scripts/monitor_v85_then_run_evals.sh` watcher            |
| 2146696ff | —   | `scripts/monitor_v85_evalsuite_then_launch_voxelpose.sh`   |

Because the v85 eval is actively reading `outputs/ablations/v85_random_view_dropout_medium_a800.pth`, **do not remove any v85 checkpoint or log until the eval suite finishes**.

## Read-only analysis commands used

```bash
ssh a800-D "df -h /mnt/nvme0n1p1"
ssh a800-D "du -sh /mnt/nvme0n1p1/zhangzy/*"
ssh a800-D "du -sh /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/*"
ssh a800-D "du -sh /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/*"
ssh a800-D "du -sh /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/tmp/*"
ssh a800-D "du -sh /mnt/nvme0n1p1/zhangzy/.cache/*"
ssh a800-D "ls -lh /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/*.pth"
ssh a800-D "pgrep -af 'v85|v86|train|python'"
```

## Recommended cleanup actions

### 1. Run the existing safe cleanup script (dry-run first)

```bash
ssh a800-D "cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20 && bash scripts/cleanup_a800_safe.sh --dry-run"
```

Expected behavior from existing `scripts/cleanup_a800_safe.sh`:

| Candidate                                                                 | Est. size | Notes |
|---------------------------------------------------------------------------|-----------|-------|
| `outputs/ablations/*.pth` where a corresponding `*_final.pth` exists    | ~39 MB    | Skips active v85/v86 checkpoints automatically. |
| `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.pth`     | 0 B       | Already removed or non-existent. |
| `~/.cache/pip` purge via `pip cache purge`                               | ~15 GB    | Largest single gain; pip packages can be re-downloaded. |
| `/mnt/nvme0n1p1/zhangzy/.cache/uv`                                        | check     | Only if uv is not in PATH. |

> **Caution:** The current dry-run / actual script will try to remove `v85_random_view_dropout_medium_a800.pth` because `v85_random_view_dropout_medium_a800_final.pth` exists. **Do not execute the non-dry-run script while PID 2218949 is running**, or add a guard for the active v85 eval.

### 2. Package caches (safe, large impact)

```bash
# 2.1 Purge pip HTTP cache (~15 GB)
ssh a800-D "rm -rf /mnt/nvme0n1p1/zhangzy/.cache/pip/http-v2"
ssh a800-D "rm -rf /mnt/nvme0n1p1/zhangzy/.cache/pip/http"

# 2.2 Remove warp compiler cache (~422 MB)
ssh a800-D "rm -rf /mnt/nvme0n1p1/zhangzy/.cache/warp"

# 2.3 Remove RTMPose / rtmlib download cache (~315 MB)
ssh a800-D "rm -rf /mnt/nvme0n1p1/zhangzy/.cache/rtmlib"
```

Expected space from Section 2: **~15.7 GB**.

### 3. Project-internal bytecode caches (safe, small impact)

```bash
ssh a800-D "find /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20 -type d -name '__pycache__ -o -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true"
```

Expected space: **~3 MB**.

### 4. Verify-then-delete candidates (medium risk)

| Path | Est. size | Why it might be safe | Verify first |
|------|-----------|----------------------|--------------|
| `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/tmp/sota_baselines/h36m_true_gt_baseline_format.pkl` | 621 MB | Large cached SOTA preprocessing artifact | Confirm no active SOTA baseline script reads it |
| `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/tmp/sota_baselines/voxelpose_data/h36m_true_gt_annotations.pkl` | 621 MB | Large cached SOTA preprocessing artifact | Confirm no active SOTA baseline script reads it |
| `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800_backup_epoch5.pth` | 39 MB | Redundant if v85 training/eval is done | Wait until PID 2218949 and v85 eval monitor finish |
| `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.pth` | 39 MB | Duplicate of `_final.pth` once v85 eval finishes | Wait until PID 2218949 finishes |
| `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm.bak.20260807075928` | 120 MB | Old repo backup outside iter20 tree | Confirm no one references it |
| `/mnt/nvme0n1p1/zhangzy/iter20.tar.gz` | 3.1 MB | Possibly stale archive | Confirm not needed for deployment |

If all verified safe:

```bash
ssh a800-D "rm -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/tmp/sota_baselines/h36m_true_gt_baseline_format.pkl"
ssh a800-D "rm -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/tmp/sota_baselines/voxelpose_data/h36m_true_gt_annotations.pkl"
ssh a800-D "rm -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800_backup_epoch5.pth"
ssh a800-D "rm -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.pth"
ssh a800-D "rm -rf /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm.bak.20260807075928"
ssh a800-D "rm -f /mnt/nvme0n1p1/zhangzy/iter20.tar.gz"
```

Expected additional space from Section 4 (if all verified): **~1.4 GB**.

## Summary of expected space freed

| Section | Description | Est. freed |
|---------|-------------|-----------:|
| 1 (existing script) | Duplicate checkpoints + pip cache | ~15.0 GB |
| 2 | Additional caches (warp, rtmlib, pip http) | ~0.7 GB |
| 3 | Bytecode caches | ~0.003 GB |
| 4 | Verified deletions | ~1.4 GB |
| **Total conservative** | | **~16 GB** |
| **Total if all verify-then-delete approved** | | **~17 GB** |

## Do-not-touch list

- Any file under `data/`, especially `data/h36m_true_gt/` and `data/webbridge/`.
- Any final checkpoint needed for the paper: `v25_true_gt_stability_a800_final.pth`, `v81_true_gt_h36m_medium_a800_final.pth`, `v82_true_gt_h36m_medium_a800_final.pth`, etc.
- `outputs/ablations/v85_random_view_dropout_medium_a800_final.pth` while post-training eval is in progress or planned.
- Running processes on GPU 6 and GPU 7 (PID 2218949 and associated eval/monitors).
- `/mnt/nvme0n1p1/zhangzy/projects` and the A800 Docker `motionflow` service (read-only per project policy).

## Suggested execution order

1. Confirm v85 eval (PID 2218949) and monitor (PIDs 2072252*, 2146696*) have finished.
2. Run `scripts/cleanup_a800_safe.sh --dry-run` and review output.
3. Execute Section 2 (package caches) — safe and high-impact.
4. Execute Section 3 (bytecode caches) — safe and quick.
5. Verify Section 4 candidates one-by-one, then execute the approved `rm` commands.
6. Re-check `df -h /mnt/nvme0n1p1`.
