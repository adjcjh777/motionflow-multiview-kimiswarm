# MotionFlow-MultiView Agent Notes

This file captures the current A800-D workflow, tmux conventions, and issue labels used in the ICRA/CVPR 2027 multi-view pose project.

## A800-D status snapshot

| Run | Description | GPU | Status |
|-----|-------------|-----|--------|
| v25 full | v18 + geometry fusion (full WebBridge/H36M/MPI) | GPU4 | Running |
| v25 small | v18 + geometry fusion (small subset) | GPU7 | Epoch 1 18.31 mm (best), epoch 3 66.56 mm; early stopping likely triggered; shared GPU7 with v11 fullscale |
| v25 ablation | v18 + geometry fusion with `geom_loss_weight=1.0` | GPU6 | Running |
| v18 | v18 deformable attention baseline | GPU5 | Running (legacy baseline) |
| v11 fullscale | IRLS full-scale baseline | GPU7 | Running; shared GPU7 with v25 small |
| v26 small | v18 + temporal geometry fusion | — | Prepared; blocked until a GPU frees |
| v21 | neural BA | — | Stopped; regressed to 128.27 mm |

## Local RTX 4090 status snapshot

| Run | Description | Status |
|-----|-------------|--------|
| v25 small local | v18 + geometry fusion + v25_dropout=0.2 + train_samples=500 | Done; epoch 1 val_MPJPE 63.13 mm; early stopping or wait wrapper moved on |
| v25 + v18 top-k ST local | v25 small + deformable top-k straight-through | Stopped after epoch 1 val_MPJPE 59.53 mm; no epoch 2 val (process exited); GPU now free |
| v29c full SEH-MV local | v25 + v29 hierarchical + TTE + physical loss | Ready; launching now on RTX 4090 |
| v18 top-k ST | v18 deformable attention with straight-through top-k | Merged to main |
| v19 temporal perceiver | Feature-aware temporal Perceiver | Merged to main |
| v26 temporal gate | Warm-startable residual gate | Merged to main |
| v27 UDP | Warm-start uncertainty depth proposals | Merged to main |
| v28 physical-space alignment | Redesigned bounded-residual physical-space alignment | Merged to main |
| outlier adaptive | Adaptive learnable outlier-view thresholds | Merged to main |
| v26+UDP full | v26 + v27 UDP + early stopping + weight decay | Stopped; best epoch 2 val 44.72mm, then overfit to 122.11mm |
| v26+UDP-GMM full | v26 + v27 UDP-GMM + early stopping + weight decay | Stopped; best epoch 4 val 40.27mm, then overfit |
| v26+UDP+v28 full | v26 + v27 UDP + v28 physical-space alignment | Stopped; epoch 1 val 83.38mm, epoch 2 val 114.70mm (v28 weights too high, reduced to 0.01) |
| v26+UDP-GMM+v28 full | v26 + v27 UDP-GMM + v28 physical-space alignment | Stopped; epoch 1 val 78.57mm, epoch 2 val 121.97mm |
| v13 temporal | Legacy v13 temporal run | Stopped to free GPU for v26 full queue |
| v29c fast SEH-MV local | v29 full SEH-MV with fast config (clip_len=9, train_samples=100) | Running on RTX 4090; first-epoch val_MPJPE pending |

### v29 SEH-MV A800-D runs

| Run | Description | GPU | Status |
|-----|-------------|-----|--------|
| v29 full SEH-MV | v25 + v29 hierarchical + TTE + physical loss (batch 24, d=128) | GPU5 | Running; first-epoch val_MPJPE pending |
| v29a | hierarchical encoder only (no TTE, no physical) | GPU1 | Running; first-epoch val_MPJPE pending |
| v29b | hierarchical + TTE (no physical) | GPU2 | Running; first-epoch val_MPJPE pending |
| v29d | TTE + physical (no hierarchical) | GPU3 | Running; first-epoch val_MPJPE pending |

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

## Long-running local runs (tmux is not available in the WSL env)

Use the nohup-based launcher to keep the training queue alive after the shell disconnects:

```bash
bash scripts/nohup_run_v26_full_queue_local_4090.sh
```

Monitor the background process with:

```bash
tail -f outputs/v26_full_queue_local_4090_nohup.log
```

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
