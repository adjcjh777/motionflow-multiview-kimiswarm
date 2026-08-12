# MotionFlow-MultiView Project Status Dashboard

> **DEPRECATED:** See the latest dashboard at [`docs/status_dashboard_v2.md`](status_dashboard_v2.md) (last refreshed 2026-08-12 ~07:55 UTC).

> **Last updated:** 2026-08-12 02:38 UTC  
> **Local repo:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **A800 training repo:** `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`  
> **Status snapshot:** AIST++ fast v2 healthy on GPU 5; mixed H36M+AIST++ stable on GPU 6; MPI RTMPose detection 10/16 on GPU 7; v25 variable-view DLT-fallback re-eval running on GPU 4.

---

## 1. A800 GPU Status (GPU 4–7)

| GPU | Utilization | Memory Used | State | Occupant |
|----:|------------:|------------:|-------|----------|
| 4 | 34 % | 811 / 81920 MiB | RUNNING | v25 variable-view DLT-fallback re-eval (PID `628743`) |
| 5 | 92 % | 19599 / 81920 MiB | RUNNING | AIST++-only fast v2 (PID `117455`) |
| 6 | 86 % | 33101 / 81920 MiB | RUNNING | v25 H36M + AIST++ mixed-dataset stability (PID `175675`) |
| 7 | 59 % | 14808 / 81920 MiB | RUNNING | MPI-INF-3DHP RTMPose detection (PID `2527668`) + CPU watcher (PID `3779633`) |

*GPU 0–3 host VLLM workers; do not touch.*

---

## 2. Active Runs

| PID | GPU | Task | Log Path | Status |
|----:|----:|------|----------|--------|
| `628743` | 4 | v25 var-view DLT-fallback re-eval | `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback_nohup.log` | RUNNING (`num_subsets_per_k=20`) |
| `117455` | 5 | AIST++-only medium fast v2 | `outputs/ablations/aistpp_only_medium_a800_fast_v2.log` | RUNNING; train loss ~27.1 @ step 2300 |
| `175675` | 6 | v25 H36M+AIST++ mixed-dataset stability | `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log` | RUNNING; train loss ~29.2 @ step 3300 |
| `2527668` | 7 | MPI-INF-3DHP RTMPose 2D detection | `outputs/mpi_rtmpose_detected_2d/generate_20260811_191500.log` | RUNNING; **10/16** `.npz` complete |
| `3779633` | — | MPI DLT-baseline CPU watcher | `outputs/mpi_rtmpose_detected_2d/wait_and_run_dlt_*.log` | WAITING for 16/16 `.npz` |

### Run Details

- **GPU 4 – v25 var-view DLT-fallback re-eval**
  - Command: `experiments/eval_variable_views.py` on `v25_true_gt_stability_a800.pth` with `--var_view_dlt_fallback`.
  - Smoke results (S9 act 02/14): k=2 **100.17 mm**, k=3 **30.72 mm**, k=4 **111.33 mm**.
  - Final output pending: `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json`.

- **GPU 5 – AIST++-only fast v2**
  - Epoch 1 val MPJPE **91.43 mm**; training healthy, loss decreasing.
  - Latest sampled train loss: ~27.1 @ step 2300.
  - Checkpoint: `outputs/ablations/aistpp_only_medium_a800_fast_v2.pth`.

- **GPU 6 – v25 H36M+AIST++ mixed stability**
  - Epoch 1 val MPJPE **36.49 mm**; Epoch 2 val **52.27 mm** reported earlier; no divergence observed.
  - Latest sampled train loss: ~29.2 @ step 3300.
  - Checkpoint: `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.pth`.

- **GPU 7 – MPI-INF-3DHP RTMPose detection**
  - Progress: **10/16** `.npz` files written to `data/webbridge/mpi_inf_3dhp_detected_2d/`.
  - CPU watcher will auto-run DLT baseline once all 16 files are ready.

---

## 3. H36M True-GT Leaderboard (S1,5,6,7,8 → S9/S11)

| Method | Test MPJPE (mm) | PA-MPJPE (mm) | Notes |
|--------|----------------:|---------------:|-------|
| Iskakov ICCV 2019 | **23.40** | 23.15 | Best val 23.35 @ epoch 9 |
| DLT (conf-weighted) | **25.67** | 25.55 | Frozen ref |
| RANSAC/conf-DLT | **26.47** | — | Frozen ref |
| **v25 stability** | **30.83** weighted / 31.56 weighted official | **33.59** | Best learned result; best val 31.13 @ Epoch 10 |
| v25 mixed H36M+AIST++ | **33.42** | 34.60 | Diverged @ Epoch 3 |
| v81 temporal-pose-attention | **37.83** | 37.75 | Completed 8 epochs |
| v82 multi-scale temporal-pose-attention | **39.46** | 39.94 | Completed 8 epochs |
| v25 medium | **43.93** | — | Local RTX 4090 |
| v46 SVG | **52.46** | 40.20 | A800 |
| v80 regularization | **53.98** | 32.47 | A800 |
| v52 UWT | **54.01** | 42.22 | A800 |
| v57 re-run | **57.10** | 37.30 | A800 |

---

## 4. Key Milestones & Latest Fixes

- **H36M circular-label problem fixed:** `data/h36m_true_gt/` is non-circular (direct MJE 13–34 mm); `scripts/diagnose_circular_labels.py` confirms `direct MJE = 0.0000 mm` on old `data/h36m_hf/*.npz`.
- **Trainer checkpoint bug fixed:** Best checkpoint now monitors `mpjpe` instead of `loss`.
- **Sparse-view (k<4) temporary fix:** `--var_view_dlt_fallback` falls back to confidence-weighted DLT when `n_active < n_views_max`.
- **AIST++ NaN fix:** Converter now uses `keypoints3d_optim`, zeroes NaN 2D keypoints with confidence 0, and drops frames only if 3D joints are NaN.
- **AIST++ full DLT baseline done:** MPJPE **15.93 mm** weighted / **38.11 mm** unweighted; PA-MPJPE **21.12 mm** / **42.66 mm**.
- **Stale circular configs deprecated:** Moved to `configs/deprecated/circular/`.

---

## 5. Next Actions

### Immediate (this session)

1. **Monitor GPU 4 – v25 var-view DLT-fallback re-eval**
   - Wait for `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json`.
   - Record S9/S11 k=2/3/4 MPJPE@k curves and update `docs/results_true_gt_h36m.md`.

2. **Monitor GPU 5 – AIST++-only fast v2**
   - Watch for Epoch 2+ validation metrics.
   - Once finished/early-stopped, evaluate on H36M S9/S11 true-GT test via `scripts/eval_aistpp_only_on_h36m_test_a800.sh`.

3. **Monitor GPU 6 – v25 H36M+AIST++ mixed stability**
   - Continue tracking loss; look for divergence once validation begins.

4. **Monitor GPU 7 – MPI detection**
   - Poll until 16/16 `.npz` files are written.
   - CPU watcher will run DLT baseline automatically; record results.

### Short-term backlog

- **Fundamentally fix k<4 learned-model failure:** Investigate random view dropout or view-count-conditioned residual head; smoke locally on RTX 4090 before A800 medium run.
- **A800 disk cleanup:** `/mnt/nvme0n1p1` is 99 % full (~46 GB free). Remove old root-level `.pth` checkpoints, abandoned v83 checkpoint, and stale tarballs after active runs finish.
- **Paper rewrite:** Continue `docs/paper_draft_icra_cvpr_2027.md` with corrected true-GT numbers, sparse-view DLT-fallback results, MPI/AIST++ cross-dataset results.

---

## 6. Blockers & Watch-outs

| Priority | Item | Status |
|----------|------|--------|
| P0 | Sparse-view k<4 learned model still fails without DLT fallback | Temporary fallback in place; root-cause fix pending |
| P1 | MPI-INF-3DHP detection in progress (10/16) | Wait for completion + DLT baseline |
| P2 | A800 disk space 99 % full | Avoid large outputs; schedule cleanup |
| P3 | AIST++-only fast v2 val still high (~95.99 mm @ Epoch 2) | Continue monitoring; cross-eval on H36M once done |

---

## 7. Quick Inspection Commands

```bash
# A800 GPU overview
ssh a800-D "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"

# Var-view re-eval log
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback_nohup.log"

# AIST++ training log
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/aistpp_only_medium_a800_fast_v2.log"

# Mixed training log
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log"

# MPI detection progress
ssh a800-D "ls /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/data/webbridge/mpi_inf_3dhp_detected_2d/*.npz | wc -l"
```

---

*Auto-generated from `AGENTS.md` and `docs/handoff_qwen3.8max.md`; refreshed with live `ssh a800-D` status.*
