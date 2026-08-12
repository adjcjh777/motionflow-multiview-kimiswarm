# Handoff: Next Agent (qwen3.8max)

**Date:** 2026-08-11 22:50 UTC
**Local repo:** `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`
**A800 training repo:** `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`

---

## 1. Objective

Keep the CVPR 2027 orchestration moving on two fronts:

1. **Monitor and harvest results** from the currently running A800 jobs.
2. **Prepare and launch the next experiment(s)** once GPU capacity opens, while respecting read-only boundaries.

Do not start a new training run that collides with existing GPU reservations (see below).

---

## 2. Repository Boundaries (MUST RESPECT)

| Boundary | Access | Path |
|----------|--------|------|
| Local WSL repo | Read/Write | `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm` |
| A800 training repo | Read/Write (via `ssh a800-D`) | `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20` |
| A800 projects dir | Read-only | `/mnt/nvme0n1p1/zhangzy/projects` |
| A800 Docker `motionflow` service | Inspect-only, do not modify | `ssh a800-D` → check the container; do not change it |

---

## 3. Current A800 Runs

### 3.1 AIST++-only medium fast v2 (GPU 5)
- **PID:** `117455`
- **Log:** `outputs/ablations/aistpp_only_medium_a800_fast_v2.log`
- **GPU:** A800:1 (`00000000:B1:00.0`, ~19584 MiB)
- **Status:** RUNNING, loss decreasing monotonically
- **Last seen:** step 1750, `loss=9.37`
- **Next checkpoint:** first validation (`val_loss` / `val_MPJPE`) — poll the log.
- **Action needed:** Wait for validation numbers. Once `val_MPJPE` is finite and reasonable, consider killing or letting finish; then evaluate the checkpoint on H36M S9/S11 true-GT test.

### 3.2 v25 H36M + AIST++ mixed-dataset stability (GPU 6)
- **PID:** `175675`
- **Log:** `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log`
- **GPU:** A800:2 (`00000000:D0:00.0`, ~33088 MiB)
- **Status:** RUNNING, loss decreasing
- **Last seen:** step 2100, `loss=22.97`
- **Next checkpoint:** First validation epoch; historical mixed-dataset run diverged at Epoch 3 (val 584 mm) but best checkpoint tested at **33.42 mm** avg on H36M S9/S11. Watch for divergence.
- **Action needed:** Poll for val MPJPE; save the best checkpoint path. If it diverges, kill and use the best saved checkpoint for test evaluation.

### 3.3 MPI-INF-3DHP RTMPose detection (GPU 7)
- **PID:** `2527668`
- **Log:** `outputs/mpi_rtmpose_detected_2d/generate_20260811_191500.log`
- **GPU:** A800:3 (`00000000:D3:00.0`, ~1752 MiB)
- **Status:** RUNNING
- **Progress:** `7/16 .npz` files in `data/webbridge/mpi_inf_3dhp_detected_2d/`
- **Next checkpoint:** 16/16 `.npz` files; then run the DLT baseline on the detected 2D.
- **Action needed:** Poll the directory count; do not launch another GPU job on GPU 7 until this finishes.

### 3.4 v25 stability variable-view evaluation (GPU 4) — COMPLETE
- **Old PID:** `4184808` — no longer active
- **Outputs:**
  - `outputs/variable_view_v25_true_gt_stability_a800.json`
  - `outputs/variable_view_v25_true_gt_stability_a800.csv`
  - `outputs/variable_view_v25_true_gt_stability_a800.log`
- **Key results (MPJPE@k mm):**
  - S9: k=2 `3482.62`, k=3 `1042.45`, k=4 `116.98`
  - S11: k=2 `3376.04`, k=3 `1030.19`, k=4 `110.58`
- **Action needed:** Copy numbers into `docs/results_true_gt_h36m.md`; note k=2/k=3 remain catastrophic.

### 3.5 Iskakov ICCV 2019 learnable-triangulation baseline (GPU 4) — COMPLETED
- **PID:** `222003`
- **Log:** `outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu4.log`
- **GPU:** A800:4 (`00000000:AD:00.0`)
- **Status:** COMPLETED (GPU 4 now free)
- **Results:** best val **23.40 mm** @ epoch 9 (S9 27.15 mm, S11 19.65 mm)
- **Action needed:** Record in `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md` (done).

### 3.6 v25 stability variable-view re-evaluation (GPU 4) — RUNNING
- **PID:** `264361`
- **Log:** `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_fixed_nohup.log`
- **GPU:** A800:4
- **Status:** RUNNING
- **Purpose:** Re-evaluate v25 stability with the fixed `variable_view_inference.py` wrapper (explicit `view_mask`).
- **Action needed:** Wait for `.csv/.json` outputs and compare k=2/k=3 numbers to the old catastrophic values.

---

## 4. GPU Map

| GPU | PID | Task | Notes |
|-----|-----|------|-------|
| 0-3 | — | VLLM workers | RESERVED — do not touch |
| 4 | 264361 | v25 var-view re-eval (wrapper fix) | H36M true-GT S9/S11, k=2/3/4; running |
| 5 | 117455 | AIST++ only | Loss decreasing to ~9.4 @ step 1750; no val yet |
| 6 | 175675 | v25 mixed H36M+AIST++ | Step ~2100, loss ~23.0; watch for divergence |
| 7 | 2527668 | MPI detection | 7/16 .npz done |

GPU 4 is now running the v25 variable-view re-evaluation.

---

## 5. Immediate Next Actions (in order)

1. **Poll AIST++ training every 10-15 min until first validation.**
   ```bash
   ssh a800-D "tail -n 30 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/aistpp_only_medium_a800_fast_v2.log"
   ```
2. **Poll mixed-dataset training for divergence/val MPJPE.**
   ```bash
   ssh a800-D "tail -n 30 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log"
   ```
3. **Poll MPI detection progress.**
   ```bash
   ssh a800-D "ls /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/data/webbridge/mpi_inf_3dhp_detected_2d/*.npz | wc -l"
   ```
4. **Once AIST++ training finishes or yields a reasonable best checkpoint:**
   - Evaluate it on H36M S9/S11 true-GT test.
   - If good, launch a H36M + AIST++ mixed-dataset run if not already running.
5. **Once mixed-dataset run converges/diverges:**
   - If it converges, run test eval on H36M S9/S11 and compare to v25 stability (**31.56 mm** weighted baseline).
   - If it diverges, kill and test the best saved checkpoint.
6. **Once MPI detection hits 16/16 .npz:**
   - Run the DLT baseline on detected 2D and record results.
7. ✅ **Monitor Iskakov baseline on GPU 4** — COMPLETED. Best val **23.40 mm** @ epoch 9.
8. **Wait for the v25 variable-view re-evaluation (wrapper fix) to finish.**
   - Running on GPU 4 (`PID 264361`).
   - Outputs will land in `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_fixed.{csv,json}`.
   - Update `docs/results_true_gt_h36m.md` with corrected k=2/k=3 numbers.

---

## 6. Files to Watch / Update

- `docs/results_true_gt_h36m.md` — add v25 var-view numbers and any new results.
- `docs/results_true_gt_shelf_campus.md` — update only if new runs touch Shelf/Campus.
- `AGENTS.md` — refresh status if significant events occur.

---

## 7. Blockers / Warnings

- **GPU 0-3 are reserved for VLLM.** Do not schedule training there.
- **A800 disk:** `/mnt/nvme0n1p1` was 99% full earlier. Avoid large checkpoint dumps; clean old outputs if needed with explicit confirmation.
- **Mixed-dataset run history:** the previous v25 mixed-dataset run diverged at Epoch 3. Keep close watch on the new stability run.
- **MPI detection is slow:** ~9+ hours elapsed for 6/16 files. Continue polling; do not block on it.
- **Sparse-view (k=2/k=3) failure root-caused and fixed:** `variable_view_inference.py` wrappers zeroed inactive-view observations but did not pass an explicit `view_mask` to `OmniMultiViewFusionV5`. Fix applied; local re-evaluation running. A800 re-run pending a free GPU.

---

## 8. Quick Command Reference

```bash
# AIST++ log
ssh a800-D "tail -n 30 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/aistpp_only_medium_a800_fast_v2.log"

# Mixed log
ssh a800-D "tail -n 30 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log"

# Active GPU jobs
ssh a800-D "nvidia-smi --query-compute-apps=timestamp,pid,gpu_bus_id,used_memory --format=csv"

# MPI progress
ssh a800-D "ls /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/data/webbridge/mpi_inf_3dhp_detected_2d/*.npz | wc -l"

# Var-view results
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v25_true_gt_stability_a800.json"
```
