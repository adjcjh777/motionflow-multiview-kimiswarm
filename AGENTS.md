# MotionFlow-MultiView Agent Notes

> **Status: DATA FOUNDATION BLOCKER — 2026-08-10**
>
> H36M labels are circular (`joints_3d == DLT(points_2d, cameras)`). All new GPU training is paused. See [`docs/data_foundation_blocker.md`](docs/data_foundation_blocker.md).
>
> Target: **CVPR 2027** (~2026-11). ICRA 2027 is too tight.

## Why we paused

- `scripts/diagnose_circular_labels.py` confirms `direct MJE = 0.0000 mm` on `data/h36m_hf/*_multiview.npz`.
- `motionflow_mv/data/webbridge_loader.py:182` triangulates the input 2D and stores it as the 3D label.
- v25–v79 numbers are therefore measuring how closely a network reproduces the DLT layer, not pose accuracy.
- The raw pkl `h36m_sh_conf_cam_source_final.pkl.zip` only contains `joint3d_image` (image-space `(u,v,z)`), which cannot be converted to a consistent world 3D across cameras.

## What is needed before resuming model work

1. **True H36M 3D GT** — either obtain the original Human3.6M mocap world coordinates, or pivot to a different dataset.
2. **Regenerate canonical `.npz`** with non-circular labels.
3. **Re-run baselines** (DLT, v25, v46, v57) on the corrected protocol.
4. **Re-orient the paper contribution** around sparse-view / cross-domain robustness, not absolute MPJPE records.

## Current data sources

| Source | True 3D? | Status |
|--------|----------|--------|
| `data/h36m_hf/*.npz` | No | Circular labels; do not use for model selection |
| `data/webbridge/h36m*.npz` | No | Same circular labels |
| MPI-INF-3DHP | Yes | Not fully downloaded; standard protocol needs detected 2D |
| Shelf/Campus | Yes | Small; inputs are GT projections |
| A800-D `/mnt/nvme0n1/zhangzy/projects` | No true H36M found | Read-only only |

## CVPR 2027 plan

| Phase | Work | Estimate |
|-------|------|----------|
| 1 | Fix data foundation (true H36M 3D or pivot dataset) | 1–2 weeks |
| 2 | Rebuild baselines on correct protocol | 3–5 days |
| 3 | Add standard SOTA comparisons (Iskakov, VoxelPose, etc.) | 1–2 weeks |
| 4 | Ablation / robustness / cross-dataset evaluation | 1–2 weeks |
| 5 | Rewrite paper with real citations and tables | 2 weeks |
| 6 | MPI official server submission + buffer | 1 week |

## GPU usage rules

- **No new training until the data foundation is fixed.**
- Local WSL + RTX 4090 is reserved for data diagnostics and smoke tests only.
- A800-D is read-only / inspection-only until corrected labels are ready.

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
