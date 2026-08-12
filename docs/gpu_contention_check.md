# GPU Contention Check

**Date/Time (local):** 2026-08-11 13:26+08  
**GPU:** NVIDIA GeForce RTX 4090 (Driver 595.79, CUDA 13.2)

## Summary

GPU is **busy** — not idle. There are **three active GPU processes** on the RTX 4090 and **one CPU-only monitoring script**. I did not terminate any process because all GPU workloads appear intentional, but they should be reconciled with the current work plan.

## `nvidia-smi` snapshot

```
Tue Aug 11 13:26:08 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 595.79                 Driver Version: 595.79  CUDA Version: 13.2 |
+-----------------------------------------+------------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage |
|=====================================+==========================|
|   0  NVIDIA GeForce RTX 4090      WDDM  |   00000000:01:00.0 Off |
| 45%   55C    P2            147W / 450W |   14084MiB / 24564MiB  |
|                                     |      GPU-Util 91%        |
+-----------------------------------------+------------------------+
```

- **GPU utilization:** 91%
- **Memory used:** 14,084 MiB / 24,564 MiB (≈57%)
- **Power draw:** 147 W / 450 W

## Active GPU (CUDA/compute) processes

| PID    | Type | Command |
|--------|------|---------|
| 19076  | C+G  | `D:\anaconda3\python.exe scripts/generate_mpi_detected_2d_from_avi.py --raw_dir data/webbridge/mpi_inf_3dhp/raw --input_dir data/webbridge/mpi_inf_3dhp --output_dir data/webbridge/mpi_inf_3dhp_detected_2d --model models/mediapipe/pose_landmarker_full.task --detect_size 384 --workers 1 --subjects 1 --seqs 1 --max_frames 3000 --chunk_size 1500` |
| 46320  | C    | `D:\anaconda3\python.exe -u experiments/train_omniview_fusion_v5_webbridge_multi.py ... --output outputs/omniview_fusion_v57_h36m_true_gt_medium.pth` (v57 H36M true-GT medium training) |
| 46960  | C    | `D:\anaconda3\python.exe -u experiments/eval_variable_views.py --model_class omniview_v5 --checkpoint outputs/omniview_fusion_v25_h36m_true_gt_medium.pth --config outputs/omniview_fusion_v25_h36m_true_gt_medium.config.json --dataset data/h36m_true_gt/s_09_acts_02_14_multiview_m.npz ... --output_json outputs/eval_variable_views_h36m_true_gt/v25_results.json` |

## Other related processes (CPU-only)

| PID    | Command |
|--------|---------|
| 43320  | `D:\anaconda3\python.exe D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm/tmp/monitor_v57.py` |

## Contention assessment

1. **Primary workload (intentional):** PID 46320 — v57 H36M true-GT medium training. This is expected given the current phase.
2. **Secondary GPU workload (questionable):** PID 46960 — variable-view evaluation of the v25 checkpoint on H36M true-GT S9. Running evaluation concurrently with training may slow the active training run and risks out-of-memory errors on a single RTX 4090.
3. **Background preprocessing (questionable):** PID 19076 — MPI-INF-3DHP detected-2D generation. It holds GPU memory (`C+G`) because MediaPipe runs on the GPU. Unless this is an urgent preprocessing step, it is likely an unneeded background GPU consumer at this time.
4. **Monitor script (benign):** PID 43320 — `monitor_v57.py` is CPU-only and appears to be tracking the training run; not a GPU contention issue.

## Recommendations

- **Do not start any new GPU training/evaluation job** until PID 46320 finishes or is explicitly stopped.
- If the v25 variable-view evaluation (PID 46960) or the MPI-INF-3DHP preprocessing (PID 19076) were started unintentionally, consider pausing/terminating them to free GPU memory for the v57 training.
- Re-check `nvidia-smi` again after any cleanup; utilization should drop when only the intended v57 training remains.

## Commands used

```bash
nvidia-smi
ps -ef | grep -iE "python|train" | grep -v grep
wmic process where "name='python.exe'" get ProcessId,CommandLine /format:csv
```
