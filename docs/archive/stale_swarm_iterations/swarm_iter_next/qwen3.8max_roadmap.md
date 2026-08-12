# qwen3.8max Agent Roadmap — True-GT H36M Divergence Ablations

**Status:** running on A800 (GPU 4 / GPU 6)  
**Last refreshed:** 2026-08-11 09:08 UTC  
**Owner:** `qwen3.8max`  

## Current state

| Run | GPU | Config key diffs | Latest val MPJPE | Artifact path (A800) |
|---|---|---|---|---|
| `v25_true_gt_baseline_fix` | 4 | `joint_limit_weight=0.0`, `temporal_bone_weight=0.0` | **46.53 mm** @ Epoch 1 | `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_baseline_fix.pth` |
| `v25_true_gt_geometry_regularization_a800` | 6 | `joint_limit_weight=0.01`, `temporal_bone_weight=0.005` | **46.75 mm** @ Epoch 1 | `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_geometry_regularization_a800.pth` |

Both jobs are alive as of last `nvidia-smi`: GPU 4 at 100% util / 66.2 GB mem, GPU 6 at 74% util / 33.1 GB mem.

## Immediate next steps (this week)

1. **Finish the two running v25 true-GT ablations.**
   - Wait for both to hit `early_stopping_patience=3` or Epoch 20.
   - Run final eval on H36M test S9/S11 for both checkpoints.
   - Record test MPJPE and best-epoch val MPJPE in `docs/results_true_gt_h36m.md`.

2. **Compare and decide the next medium run target.**
   - If geometry-regularization matches/beats baseline → queue `v25_true_gt_mixed_dataset` (H36M + MPI-INF-3DHP true-GT mixed loader) on the freed GPU.
   - If baseline wins cleanly → prepare a `v57_true_gt_fixed_trainer` medium run using the MPJPE-monitored checkpoint fix.

## Secondary queue (after v25 divergence result)

3. **Re-run v57 with the fixed trainer.**
   - Trainer now monitors `mpjpe` for best-checkpoint selection; the true best epoch will be saved.
   - Target: beat stale saved best 82.19 mm and recover the lost 75.16 mm @ Epoch 3.

4. **Replace MPI detector or complete AIST++ full-medium integration.**
   - **P0 (MPI):** Replace MediaPipe with RTMPose/CPN/HRNet trained on MPI-INF-3DHP; re-run DLT until ~20–30 mm before learned benchmarking. RTMPose batch-dim bug in `scripts/generate_mpi_detected_2d.py` is already fixed.
   - **P2 (AIST++ fallback):** Run full AIST++ medium for v25/v80/v57 using existing smoke manifests and append results.

## Repository hygiene

- 119 git branches exist locally/remotely; no merged stale branches were detected against `main` at this refresh, but the local `+` branches should be reviewed and pruned when their experiments are absorbed.
- On completion of each run, merge experiment configs/scripts into `main` and delete the corresponding experiment branch.

## Definition of done for this roadmap

- [ ] v25 baseline and geometry-regularization ablations complete with test S9/S11 MPJPE.
- [ ] Decision recorded: next medium run is either mixed-dataset v25 or v57 fixed-trainer.
- [ ] MPI detector replacement reaches DLT ≤ 30 mm OR AIST++ full-medium results appended.
- [ ] `docs/results_true_gt_h36m.md` updated with all new numbers.
