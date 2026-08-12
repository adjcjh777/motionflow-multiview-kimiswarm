# MotionFlow-MultiView A800 Status Dashboard

> Last updated: 2026-08-12 ~15:15 UTC

## Active A800 Processes

| PID | GPU | Type | Command | Notes |
|------|------|------|---------|-------|
| 2218949 | 6 | v85 no-fallback eval | `experiments/eval_variable_views.py` | Running; outputs for k=2/3/4 exist |
| 1921455 | 6 | other project | `.venv-cu130-a800` | GPU policy violation |
| 20857 | 7 | LuxTTS | `/home/LuxTTS/.venv/bin/python3` | GPU policy violation |
| 20880 | 7 | Mega-ASR | `/home/Mega-ASR/.venv/bin/python3` | GPU policy violation |
| 1853395 | 7 | other project | `.venv-cu130-a800` | GPU policy violation |

## GPU Utilization (0-7)

| GPU | Util | Memory Used | Status |
|-----|------|-------------|--------|
| 0 | 0% | 76135 MiB | VLLM (reserved) |
| 1 | 0% | 76135 MiB | VLLM (reserved) |
| 2 | 0% | 76135 MiB | VLLM (reserved) |
| 3 | 0% | 76135 MiB | VLLM (reserved) |
| 4 | 0% | 673 MiB | other |
| 5 | 0% | 57847 MiB | VLLM EngineCore (reserved) |
| 6 | 35% | 1481 MiB | MotionFlow eval + other |
| 7 | 0% | 13051 MiB | other projects (violation) |

## Latest Results

### v85 no-fallback variable-view eval (complete outputs)

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | 2310.27 | 1119.45 | 83.52 |
| S11 | 2308.80 | 1118.18 | 77.07 |

- k<4 remains catastrophic (~1100-2300 mm) for the learned v85 model without DLT fallback.
- k=4 is reasonable (S9 83.52 / S11 77.07 mm) but worse than v82 k=4 (47.81 / 42.36 mm).

### v2 baselines

- DLT (conf-weighted): 25.67 mm
- RANSAC/conf-DLT: 26.47 mm

## Disk

- `/mnt/nvme0n1p1`: 99% used, ~58 GB free (critical)

## Blockers

1. GPU 6/7 violations: other projects using MotionFlow-reserved GPUs.
2. v85 random view dropout did not solve k<4 catastrophic failure.
3. A800 disk critically low.

## Next Actions

1. Wait for v85 DLT-fallback eval (if queued) or run it once GPU 6 is free.
2. Run `scripts/cleanup_a800_safe.sh` dry-run to identify safe deletions.
3. Sync v2 labels and restart learned leaderboard once GPU 6/7 are free and clean.
