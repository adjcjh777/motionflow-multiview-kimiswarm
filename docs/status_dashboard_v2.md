# MotionFlow-MultiView Status Dashboard

> Updated: 2026-08-12 ~10:45 UTC
> GPU policy: A800 GPUs 6/7 only; GPUs 0-5 reserved.

## Active runs (A800)

| GPU | PID(s) | Task | Status | Latest numbers |
|-----|--------|------|--------|----------------|
| 6 | 1649421 | v25 stability var-view DLT-fallback | RUNNING | — |
| 6 | 1649422 | v81 temporal-pose-attention var-view DLT-fallback (k=2,3) | **COMPLETED** | S9 58.18/33.32 mm; S11 49.35/25.28 mm (k=4 not run) |
| 6 | 1604077 | v82 multi-scale temporal-pose-attention var-view DLT-fallback | **COMPLETED** | S9 58.18/33.32/47.81 mm; S11 49.35/25.28/42.36 mm |
| 7 | 1653903 | v85 random view dropout medium | RUNNING | Epoch 5 val MPJPE 39.32 mm |

## GPU utilization (0-7)

```
index, utilization.gpu [%], memory.used [MiB]
0, 100 %, 76135 MiB  # VLLM
1, 100 %, 76135 MiB  # VLLM
2, 100 %, 76135 MiB  # VLLM
3, 100 %, 76135 MiB  # VLLM
4, 0 %, 425 MiB      # free
5, 0 %, 641 MiB      # free
6, 33 %, 2040 MiB    # v25/v81/v82 evals
7, 89 %, 46146 MiB   # v85 training
```

## Disk

- `/mnt/nvme0n1p1`: 99% full, ~42 GB free.

## Completed milestones

- MPI-INF-3DHP RTMPose detection 16/16 done; DLT baseline MPJPE **115.09 mm**, PA-MPJPE **132.68 mm**.
- AIST++-only fast v2 → H36M cross-eval done: S9 **98.17 mm**, S11 **89.70 mm**, combined **93.94 mm**.
- v82 var-view DLT-fallback completed with numbers above.
- GPU policy updated: only GPUs 6/7 used; 4/5 vacated.

## Next actions

1. Wait for v25/v81 var-view DLT-fallback JSONs and record k=2/3/4 numbers.
2. Wait for v85 to finish training, then launch variable-view eval (with/without DLT fallback).
3. Continue monitoring disk usage; run cleanup if free space drops below 30 GB.
