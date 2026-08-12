# Turn Summary — 2026-08-12

**Scope:** Refresh of A800 state, current runs, and key numbers for the MotionFlow-MultiView project.  
**File:** `docs/turn_summary_2026-08-12.md`

---

## 1. What was launched / achieved this round

| Item | Status | What happened |
|------|--------|---------------|
| **v85 random view dropout (H36M true-GT medium)** | `RUNNING` on A800 GPU 4 | Launched to address the k<4 sparse-view structural failure. Uses `use_random_view_dropout_v85`, dropout prob 0.3, min 2 views, count embedding. |
| **v82 variable-view DLT-fallback eval** | `RUNNING` on A800 GPU 5 | S9 (PID 1461193) and S11 (PID 1541733) evals launched; no output JSONs yet, logs still empty. |
| **v81 variable-view DLT-fallback eval** | `RUNNING` on A800 GPU 5 | S9 eval running (PID 1518125); S11 log file exists but was not observed as an active process at inspection time. |
| **AIST++-only → H36M cross-eval** | `COMPLETED` | S9 **98.17 mm**, S11 **89.70 mm**; average **93.94 mm**, weighted **94.71 mm**. Output: `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`. |
| **MPI-INF-3DHP RTMPose DLT baseline** | `COMPLETED` | Mean MPJPE **115.09 mm**, mean PA-MPJPE **132.68 mm** across 16/16 `.npz` files. Output: `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`. |

No new long training runs were launched by this agent; only inspection commands were run.

---

## 2. Current A800 GPU occupancy

| GPU | Model | Memory | Util. | Occupant |
|-----|-------|--------|-------|----------|
| 0 | NVIDIA A800-SXM4-80GB | 76.1 / 81.9 GB | 100 % | VLLM worker (Qwen3.6-27B, 4-way TP) |
| 1 | NVIDIA A800-SXM4-80GB | 76.1 / 81.9 GB | 100 % | VLLM worker |
| 2 | NVIDIA A800-SXM4-80GB | 76.1 / 81.9 GB | 100 % | VLLM worker |
| 3 | NVIDIA A800-SXM4-80GB | 76.1 / 81.9 GB | 100 % | VLLM worker |
| 4 | NVIDIA A800-SXM4-80GB | 33.8 / 81.9 GB | 87 % | **v85 random view dropout training** (PID 1370214) |
| 5 | NVIDIA A800-SXM4-80GB | 2.3 / 81.9 GB | 100 % | v81 + v82 variable-view DLT-fallback evals |
| 6 | NVIDIA A800-SXM4-80GB | 1.2 / 81.9 GB | 31 % | Mostly free |
| 7 | NVIDIA A800-SXM4-80GB | 13.1 / 81.9 GB | 0 % | MPI residual memory, not fully released |

- **GPU 4** is busy with v85; do not touch.
- **GPU 5** is the only currently usable slot for non-VLLM work, but it is already running the v81/v82 evals.
- **GPU 6** is mostly idle (1.2 GB); could host lightweight evals if scheduled.
- **GPU 7** still holds ~13 GB from the MPI run; may need a cleanup/restart before a large training job uses it.

---

## 3. Key numbers

### v85 random view dropout (in progress)
- **PID:** 1370214
- **Start:** 2026-08-12 06:26:20 UTC
- **Latest epoch logged:** Epoch 2 (still running)
- **Val MPJPE so far:**
  - Epoch 1: **62.85 mm**
  - Epoch 2 (mid-epoch): **36.90 mm**
- **Checkpoint/log:** `outputs/ablations/v85_random_view_dropout_medium_a800.*`

### AIST++-only → H36M cross-eval
- S9: **98.17 mm** MPJPE, **49.44 mm** PA-MPJPE
- S11: **89.70 mm** MPJPE, **39.55 mm** PA-MPJPE
- Average: **93.94 mm**
- Frame-weighted: **94.71 mm**

### MPI-INF-3DHP RTMPose DLT baseline
- Mean MPJPE: **115.09 mm**
- Mean PA-MPJPE: **132.68 mm**
- Per-file range: 84.2 mm – 162.7 mm MPJPE

### v81 / v82 variable-view DLT-fallback
- **v82 S9** (PID 1461193, started 06:58): running, no output yet.
- **v82 S11** (PID 1541733, started 07:31): running, no output yet.
- **v81 S9** (PID 1518125, started 07:20): running, no output yet.
- **v81 S11** log file exists (`outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_S11.*`) but no active process was observed at inspection time.

---

## 4. Top 3 risks / blockers

1. **Sparse-view (k<4) fix still unvalidated.**  
   v85 is the first training run designed to natively handle k=2/3/4, but it is only at Epoch 2. The v81/v82 DLT-fallback evals are running but have not produced JSONs yet, so we do not know whether temporal-attention variants behave differently from v25 under the same fallback. If v85 does not close the gap with the v25 DLT-fallback baseline (S9 58.18/33.32/116.98 mm for k=2/3/4), the architecture still lacks a real sparse-view solution.

2. **A800 disk is 99 % full.**  
   `/mnt/nvme0n1p1` has only **~46 GB free** out of 3.5 TB. v85 is actively writing checkpoints and logs, and the pending variable-view evals will add more outputs. There is a real risk of jobs failing due to disk exhaustion before they finish. A dry-run of `scripts/cleanup_a800_safe.sh` should be considered once any current run completes.

3. **GPU scheduling is tight and fragmented.**  
   GPUs 0–3 are locked by VLLM. GPU 4 is occupied by v85. GPU 5 is already running v81/v82 evals. GPU 7 still carries ~13 GB of MPI memory. This leaves little headroom for new medium/long training jobs; any additional run should either queue behind v85 or carefully target GPU 6/7 after memory cleanup. Launching another training run now would collide with existing work.

---

## 5. Next immediate actions

- Wait for v85 to finish training, then run variable-view eval with and without `--var_view_dlt_fallback`.
- Monitor v81/v82 DLT-fallback eval outputs in `outputs/variable_view_fix/`.
- Run `scripts/cleanup_a800_safe.sh --dry-run` once a slot is free to free the 99 % disk before starting new training.
