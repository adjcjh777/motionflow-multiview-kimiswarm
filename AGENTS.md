# MotionFlow-MultiView Agent Notes

This file captures the current A800-D workflow, tmux conventions, and issue labels used in the ICRA/CVPR 2027 multi-view pose project.

## A800-D status snapshot

| Run | Description | GPU | Status |
|-----|-------------|-----|--------|
| v25 full | v18 + geometry fusion (full WebBridge/H36M/MPI) | GPU4 | Running, ~3,450/18,750 steps, first epoch in progress |
| v25 small | v18 + geometry fusion (small subset) | GPU7 | Running, ~2,400/3,750 steps, first epoch in progress |
| v25 ablation | v18 + geometry fusion with `geom_loss_weight=1.0` | GPU6 | Running, ~2,300/3,750 steps, first epoch in progress |
| v18 | v18 deformable attention baseline | GPU5 | Running (legacy baseline) |
| v26 small | v18 + temporal geometry fusion | — | Prepared; blocked until a GPU frees |
| v21 | neural BA | — | Stopped; regressed to 128.27 mm |

- **Remote host:** `a800-D` (SSH)
- **Remote repo:** `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`
- **Access rule:** A800-D is read-only for status checks and log inspection. Do not create, modify, or delete files there.
- **Local repo:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`
- **Hardware:** local WSL + RTX 4090 for smoke tests and fast iterations; A800-D for full runs.

## A800 workflow

1. Smoke every new run on the local RTX 4090 first (or on a lightly loaded A800 GPU with a smoke config).
2. Launch full runs in named `tmux` sessions so they survive SSH disconnects.
3. Use `scripts/auto_eval_when_ready.sh` (or its host-side cron) for periodic evaluation; do not run it manually on A800-D.
4. Check A800-D status read-only via SSH, then update the status table in this file or the relevant `docs/swarm_iter*/status.md`.
5. When a GPU frees, launch the prepared next run (e.g., `v24`) after double-checking its checkpoint path and YAML config.

## tmux usage

- **List sessions:**
  ```bash
  tmux ls
  ```
- **Attach to a session:**
  ```bash
  tmux attach -t v23_gpu4
  ```
- **Detach from a session:** `Ctrl-b` then `d`.
- **Naming convention:**
  - `<variant>_gpu<g>` for a single-GPU run, e.g. `v23_gpu4`.
  - For multi-GPU or split runs, append a short tag: `v23_split_gpu4_gpu6`.
- **Log inspection inside a session:**
  ```bash
  tail -f outputs/v23_kap_no_ba.log
  ```
- **Monitor running experiments without attaching:**
  ```bash
  ssh a800-D "tmux capture-pane -pt v23_gpu4 -S -100"
  ```
- **Do not kill a session unless the run is confirmed dead or stopped by the team.**

## Issue labels

Use the following labels on GitHub Issues / PRs:

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
