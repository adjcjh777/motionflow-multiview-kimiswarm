# MotionFlow-MultiView A800 Status Dashboard

> Last updated: 2026-08-13 ~02:30 UTC

## GitHub / Local repo status

- Remote URL token: **removed** — current origin is `https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git`
- Old worktree `.worktrees/v18_deformable_attention_baseline`: **deleted**
- Local lightweight tags `v25_local_baseline_monitor_commit` / `v25_local_baseline_monitor_v1`: **deleted**
- `main` push: **done** — commit `8aee08c` (or newer) on GitHub
- Stash backups in `patches/stashes/`: **45 patches retained** (pending audit)

## Active A800 Processes

| PID | GPU | Type | Command / Session | Notes |
|------|------|------|-------------------|-------|
| — | 6 | v25 true-GT v2 medium training | tmux `v25_true_gt_v2_medium_a800` | **DONE**; early-stop @ epoch 6; best val **31.41 mm** |
| — | 6 | v86 no-count-embedding ablation | tmux `v86_no_count_embedding` | **RUNNING**; Epoch 5/20 done, best val **31.64 mm @ Epoch 3** (Epoch 4 43.26 mm → Epoch 5 44.00 mm); Epoch 6 in progress; early stopping expected within 1–2 epochs; v2 protocol; log `outputs/ablations/v86_no_count_embedding_medium_a800.log` |
| — | 6/7 (post-v86) | v85 DLT-fallback watcher | `scripts/launch_v85_dlt_fallback_after_v86.sh` | **QUEUED**; auto-runs after v86 finishes |
| — | 7 | External project | — | **OCCUPIED**; ~12 GB; do not touch |

## GPU Utilization (0-7)

| GPU | Util | Memory Used | Status |
|-----|------|-------------|--------|
| 0 | 0% | 76135 MiB | VLLM (reserved) |
| 1 | 0% | 76135 MiB | VLLM (reserved) |
| 2 | 0% | 76135 MiB | VLLM (reserved) |
| 3 | 0% | 76135 MiB | VLLM (reserved) |
| 4 | 0% | 673 MiB | other |
| 5 | 0% | 57847 MiB | VLLM EngineCore (reserved) |
| 6 | — | — | MotionFlow v86 training (project GPU) |
| 7 | — | ~12 GB | External project (GPU policy violation) |

## Latest Results

### v25 true-GT v2 medium training

| Metric | Value |
|--------|-------|
| Best val MPJPE | **31.41 mm** |
| Best epoch | 6 |
| Early-stopped epoch | 6 |
| Checkpoint | `outputs/ablations/v25_true_gt_v2_medium_a800.pth` |
| Data protocol | `configs/splits/h36m_true_gt_v2_standard.yaml` |
| Test MPJPE | **pending** |

### v85 random-view-dropout training

| Metric | Value |
|--------|-------|
| Best val MPJPE | **31.42 mm** |
| Checkpoint | `outputs/ablations/v85_random_view_dropout_medium_a800.pth` (symlink to `..._final.pth`) |

### v85 no-fallback variable-view eval (complete outputs)

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | 2310.27 | 1119.45 | 83.52 |
| S11 | 2308.80 | 1118.18 | 77.07 |

- k<4 remains catastrophic (~1100-2300 mm) for the learned v85 model without DLT fallback.

### v85 DLT-fallback variable-view eval

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | — | — | — |
| S11 | — | — | — |

- **Queued behind v86.** Watcher `scripts/launch_v85_dlt_fallback_after_v86.sh` will auto-run the eval once v86 training finishes. No results yet.

### Latest val MPJPE (true-GT v2)

| Variant | Status | Best val MPJPE | Notes |
|---------|--------|----------------|-------|
| v25 true-GT v2 medium | ✅ DONE | **31.41 mm** | @ Epoch 6; test pending |
| v85 random-view-dropout | ✅ DONE | **31.42 mm** | @ final epoch; no-fallback var-view done |
| v86 no-count-embedding | 🔄 RUNNING | **31.64 mm** | @ Epoch 3 (best); Epoch 5/20 done (Epoch 4 43.26 mm → Epoch 5 44.00 mm); Epoch 6 in progress; early stopping expected within 1–2 epochs |

### v2 baselines

- DLT (conf-weighted): **25.67 mm**
- RANSAC/conf-DLT: **26.47 mm**

### Local smoke results (RTX 4090)

| Variant | Status | 2-epoch val MPJPE | Notes |
|---------|--------|--------------------|-------|
| v37 self-critique | ✅ DONE | **87.85 mm** | Baseline smoke. |
| v21 neural BA | ✅ FIXED | **79.42 mm** | Axis-angle rotation descriptor in `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` diverged at identity; replaced with `R - R^T` skew-symmetric part. Initial run was 93.50 mm. |
| v29 hierarchical | ✅ FIXED | **95.20 mm** | Not a bug; original smoke config was too heavy for RTX 4090. Use `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh`. |

## Disk

- `/mnt/nvme0n1p1`: ~98% used, ~72 GB free (critical)

## Blockers

1. GPU 7 occupied by an external project (~12 GB). Do not kill; do not launch MotionFlow jobs there.
2. v85 DLT-fallback eval is blocked behind v86 training (auto-triggered).
3. A800 disk critically low.

## Next Actions

1. Monitor v86 no-count-embedding ablation until completion.
2. Wait for the v85 DLT-fallback watcher to auto-trigger the eval after v86 finishes.
3. Run v25 true-GT v2 test-set evaluation on the first free project GPU.
4. Run `scripts/cleanup_a800_safe.sh` dry-run to identify safe deletions before any new large write.
