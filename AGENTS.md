# MotionFlow-MultiView Agent Notes

> **Status: H36M v25/v80 TRUE-GT MEDIUM COMPLETE, v57 RUNNING — GPU BUSY — 2026-08-11**
>
> H36M circular-label problem is resolved: true-GT `.npz` are in `data/h36m_true_gt/` and pass both reprojection and circularity gates. Baselines (DLT, Iskakov, v25, v80) are being re-run on the corrected standard protocol; A800 tmux training remains stopped.
>
> **v25 / v80 true-GT medium** are complete. v25 best val MPJPE = **72.80 mm** (epoch 2), then diverged to 207.62 mm by epoch 8. v80 best val MPJPE = **39.98 mm** (epoch 4), then overfit to 133.71 mm by epoch 8.
>
> **v57 H36M true-GT medium is currently running** on the local RTX 4090. `nvidia-smi` reports ~89% GPU utilisation and ~14.1 GB memory in use — **do not start any other GPU task until it finishes**.
>
> **MPI-INF-3DHP** remains blocked: real detected 2D is missing (no `imageSequence/` on A800 or local). **AIST++** smoke integration is complete (~44 mm DLT, smoke manifests created), but a full medium run is still pending.
>
> See the session handoffs [`docs/handoff_qwen3.8max.md`](docs/handoff_qwen3.8max.md) and [`docs/handoff_qwen3.8max_session_summary.md`](docs/handoff_qwen3.8max_session_summary.md), plus the leaderboards [`docs/results_true_gt_h36m.md`](docs/results_true_gt_h36m.md) and [`docs/results_true_gt_shelf_campus.md`](docs/results_true_gt_shelf_campus.md).
>
> Target: **CVPR 2027** (~2026-11). ICRA 2027 is too tight.

## Current work in flight

| Agent | Task | Machine | Notes |
|-------|------|---------|-------|
| `qwen3.8max` (running) | H36M true-GT v57 medium | Local RTX 4090 | Running; do not start another GPU job |
| `agent-51` (done) | H36M true-GT v25 medium | Local RTX 4090 | Completed: best 72.80 mm @ epoch 2 |
| `agent-67` (done / idle) | AIST++ smoke integration v25/v80 | Local RTX 4090 | Smoke complete; full medium pending |

- Local GPU can run **at most one training task at a time**. As of the latest check the RTX 4090 is **busy** running v57 H36M true-GT medium (`nvidia-smi` shows ~89% utilisation, ~14.1 GB memory, active `python.exe` GPU training process). If a GPU task is active, only prepare configs/scripts; do not launch a new training run.
- A800-D `/mnt/nvme0n1/zhangzy/projects` and the A800 Docker `motionflow` service are **read-only** — no writes, starts, or modifications.
- Before starting any new work, check active background tasks to confirm GPU availability.

## Why we paused

- `scripts/diagnose_circular_labels.py` confirms `direct MJE = 0.0000 mm` on `data/h36m_hf/*_multiview.npz`.
- `motionflow_mv/data/webbridge_loader.py:182` triangulates the input 2D and stores it as the 3D label.
- v25–v79 numbers are therefore measuring how closely a network reproduces the DLT layer, not pose accuracy.
- The raw pkl `h36m_sh_conf_cam_source_final.pkl.zip` only contains `joint3d_image` (image-space `(u,v,z)`), which cannot be converted to a consistent world 3D across cameras.

## Data foundation status

1. **True H36M 3D GT** — obtained; true-GT `.npz` are in `data/h36m_true_gt/`.
2. **Regenerate canonical `.npz`** with non-circular labels — done; H36M and Shelf/Campus datasets have been rebuilt.
3. **Re-run baselines** (DLT, Iskakov, v25, v46, v57, v80) on the corrected protocol — in progress. v25 H36M true-GT medium is complete (72.80 mm best); v80/v57 medium still pending. See [`docs/results_true_gt_shelf_campus.md`](docs/results_true_gt_shelf_campus.md) and [`docs/results_true_gt_h36m.md`](docs/results_true_gt_h36m.md).
4. **Re-orient the paper contribution** around sparse-view / cross-domain robustness, not absolute MPJPE records — pending.

## Current data sources

| Source | True 3D? | Status |
|--------|----------|--------|
| `data/h36m_hf/*.npz` | No | Circular labels; do not use for model selection |
| `data/webbridge/h36m*.npz` | No | Same circular labels |
| `data/h36m_true_gt/*_multiview_m.npz` | **Yes** | True mocap world coordinates; standard protocol S1,5,6,7,8 → S9/S11 |
| MPI-INF-3DHP | Yes | Labels are real mocap, but current 2D is GT-projection; standard protocol needs real detected 2D |
| Shelf/Campus | Yes | Non-circular `.npz` rebuilt from `detection.json + annotation_3d.json` at `data/webbridge/shelf_campus_detected/` |
| A800-D `/mnt/nvme0n1/zhangzy/projects` | No true H36M found | Read-only only |

## Next steps (CVPR 2027)

1. **H36M true-GT leaderboard**: v25 medium is complete (best 72.80 mm). Next run v80/v57 medium and validate v25 EMA / per-split (S9/S11) numbers.
2. **Validate true-GT numbers** in [`docs/results_true_gt_h36m.md`](docs/results_true_gt_h36m.md) and [`docs/results_true_gt_shelf_campus.md`](docs/results_true_gt_shelf_campus.md); catch any remaining circular-label leakage or protocol mismatch.
3. **Integrate AIST++** (via `agent-67`) into the v25/v80 pipeline and sanity-check 2D/3D alignment.
4. **Prepare SOTA comparison configs** for Iskakov, VoxelPose, and DLT so they can run as soon as the GPU is free.
5. **Draft the paper story** around sparse-view / cross-domain robustness, using the corrected baselines as empirical anchors.

## CVPR 2027 plan

| Phase | Work | Estimate |
|-------|------|----------|
| 1 | Close data foundation (true H36M 3D + Shelf/Campus rebuild) | 1–2 weeks |
| 2 | Rebuild baselines on correct protocol | 3–5 days |
| 3 | Add standard SOTA comparisons (Iskakov, VoxelPose, etc.) | 1–2 weeks |
| 4 | Ablation / robustness / cross-dataset evaluation | 1–2 weeks |
| 5 | Rewrite paper with real citations and tables | 2 weeks |
| 6 | MPI official server submission + buffer | 1 week |

## GPU usage rules

- **Local GPU concurrency:** RTX 4090 can run **one training task at a time**. If `agent-51` or `agent-67` is active, only prepare configs/scripts; do not launch a new training run.
- **A800 tmux training sessions are stopped.** A800-D and its Docker `motionflow` service are **read-only / inspection-only**.
- Local WSL + RTX 4090 is primarily reserved for data diagnostics, smoke tests, and baseline re-runs on the corrected protocol.

## Infrastructure

- **Remote host:** `a800-D` (SSH)
- **Remote repo:** `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`
- **Local repo:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`

### tmux

```bash
tmux ls
tmux attach -t <session>
ssh a800-D "tmux capture-pane -pt <session> -S -100"
```

### nohup (WSL)

```bash
nohup bash scripts/<script>.sh > outputs/<log>.log 2>&1 &
tail -f outputs/<log>.log
```

## Issue / PR labels

| Label | Meaning |
|-------|---------|
| `P0-blocker` | Must be resolved before the next milestone / paper deadline |
| `P1-next` | Important; pick up once P0 items are cleared |
| `P2-nice` | Useful but not urgent |
| `experiment` | New training run, model variant, or proposal |
| `ablation` | Ablation component / hyperparameter / robustness test |
| `bug` | Unexpected behavior, crash, or regression |
| `data` | Dataset, loader, pseudo-label, or preprocessing issue |
| `paper` | Writing, figure, table, or paper-story task |
| `infra` | Build, environment, A800/tmux ops, CI |
| `question` | Needs clarification or discussion |

Optional status prefixes for issue titles: `[RUNNING]`, `[STOPPED]`, `[READY]`, `[BLOCKED]`, `[DONE]`.
