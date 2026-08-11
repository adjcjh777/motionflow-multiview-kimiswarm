# Handoff: qwen3.8max → next agent (2026-08-11 21:30 UTC)

> AIST++ canonical `.npz` was contaminated with NaN; the converter has been fixed and clean data is being regenerated. v25 stability variable-view eval is running on GPU 4. MPI RTMPose detection continues on GPU 7.

## Where things stand

| Run / Task | GPU | Status | Key numbers | Output files |
|---|---:|---|---|---|
| **v25 stability variable-view eval** | 4 | **RUNNING** (PID `4184808`, started 21:07) | k=4 combined 113.78 mm previously | `outputs/variable_view_v25_true_gt_stability_a800.*` |
| **AIST++-only medium fast v2** | 5 | **STOPPED** (was PID `4021830`) — NaN val blocker | `val_loss=nan, val_MPJPE=nan` at Epoch 1 | `outputs/ablations/aistpp_only_medium_a800_fast_v2.log` |
| **MPI-INF-3DHP RTMPose detection** | 7 | **RUNNING** (PID `2527668`) | 5/16 `.npz` complete | `outputs/mpi_rtmpose_detected_2d/generate_20260811_191500.log` |
| VLLM workers | 0-3 | BUSY (not our project) | — | — |
| **GPU 6** | 6 | **IDLE** | — | — |

## Critical blocker resolved: AIST++ NaN

- **Root cause**: `convert_aistpp` in `motionflow_mv/data/webbridge_loader.py` defaulted to raw `keypoints3d`, which contains NaNs for occluded/missing joints on ~20% of sequences.
- **Fix applied** (same file): prefer `keypoints3d_optim` when available; drop any remaining NaN frames before saving canonical `.npz`.
- **Action in progress**: local WSL is regenerating `data/webbridge/aistpp_canonical/*.npz`.
- **Next step**: sync clean `.npz` to A800 and relaunch `aistpp_only_medium_a800_fast_v2` on GPU 5.

## Key numbers

| Run | GPU | Status | Best val | Test S9/S11 |
|-----|-----|--------|----------|-------------|
| v25 stability | 6 | Finished | 31.13 mm @ Epoch 10 | **31.56 mm** weighted (S9 34.87, S11 26.80), PA 34.35 |
| v81 medium | 4 | Finished | 38.62 mm @ Epoch 8 | **37.83 mm** (S9 42.19, S11 33.46), PA 37.75 |
| v82 medium | 4 | Finished | 39.58 mm @ Epoch 8 | **39.46 mm** (S9 42.07, S11 36.84), PA 39.94 |
| v57 re-run | 5 | Finished | 57.81 mm @ Epoch 4 | 57.10 mm |
| v25 mixed | 5 | Diverged @ Epoch 3 | 584.25 mm | **33.42 mm avg** (S9 37.87, S11 28.96) |
| AIST++ DLT | 6 | Finished | — | **15.93 mm** weighted / 38.11 mm unweighted |
| v81 var-view | 6 | Done | — | k=4: S9 54.53 mm, S11 47.41 mm |
| v25 var-view | 4 | Running | — | — |
| MPI RTMPose | 7 | Running | — | 5/16 `.npz` |

## What to do next

1. **Sync clean AIST++ data and restart training**
   - Wait for local conversion to finish (background task `bash-n9sde7jx`).
   - Validate no NaN remains in `data/webbridge/aistpp_canonical/*.npz`.
   - `rsync` the directory to A800 `data/webbridge/aistpp_canonical/`.
   - Relaunch `scripts/run_aistpp_only_medium_a800_gpu5_fast_v2.sh` on A800 GPU 5.
   - Verify first validation reports finite `val_loss` and `val_MPJPE`.

2. **Monitor v25 stability variable-view eval (GPU 4)**
   - Tail `outputs/variable_view_v25_true_gt_stability_a800_nohup.log`.
   - Once `.csv/.json` appear, record k=2/3/4 MPJPE@k in `docs/results_true_gt_h36m.md`.

3. **MPI RTMPose (GPU 7)**
   - Continue polling until 16/16 `.npz` are ready.
   - The CPU watcher will automatically run the DLT baseline; record the final numbers.

4. **Next experiments after AIST++-only converges**
   - Evaluate the AIST++-only checkpoint on H36M S9/S11 true-GT test.
   - If promising, launch **H36M + AIST++ mixed training** (configs already prepared in `configs/splits/h36m_aistpp_mixed.yaml` / `h36m_true_gt_aist_mixed_train_val_a800.yaml`).
   - Architecture exploration (v85+) is **deprioritized** until cross-dataset training is baselined.

5. **Repository hygiene**
   - Local and A800 Git branches have been audited; stale feature/swarm branches should be pruned (see `docs/github_branch_cleanup_audit.md` if needed).
   - Keep `AGENTS.md` and this handoff in sync after every significant state change.

## Known blockers

- **A800 disk 99% full** (~44 GB free). The clean AIST++ canonical directory is ~1.5 GB; confirm free space before syncing.
- **v82 variable-view eval** could not be confirmed running in process list; check `outputs/variable_view_v82*` and rerun if absent.

## Quick checks

```bash
# Local AIST++ conversion progress
tail -n 20 D:/KimiCodeHome/sessions/wd_motionflow-multivie-kimiswarm_391dcd43ee78/session_9c6e76c8-8be1-4d8f-b6b1-4df68b7ac0b5/agents/main/tasks/bash-n9sde7jx/output.log

# GPU availability
ssh a800-D "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv"

# v25 var-view progress
ssh a800-D "tail -n 20 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v25_true_gt_stability_a800_nohup.log"

# MPI RTMPose progress
ssh a800-D "ls /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/data/webbridge/mpi_inf_3dhp_detected_2d/*.npz 2>/dev/null | wc -l"

# Disk space
ssh a800-D "df -h /mnt/nvme0n1p1"
```

## Which GPU to use next

- **GPU 5** after AIST++ clean data sync — restart AIST++-only training.
- **GPU 6** is idle — available for the next medium run (e.g., mixed H36M+AIST++ training or a SOTA baseline).
