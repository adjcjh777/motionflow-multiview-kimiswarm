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
| v29c fast SEH-MV local | v29 full SEH-MV with fast config (clip_len=9, train_samples=100) | Killed; hung during first eval (TTE too expensive) |
| v29e fast SEH-MV local | v29 hierarchical + physical loss, no TTE (clip_len=9, train_samples=100) | Killed; first-epoch val_MPJPE=91.38mm (small config unstable) |
| v25 baseline fast local | v25 geometry fusion only, d=64, n_st_layers=2 (clip_len=9, train_samples=100) | Killed; first-epoch val_MPJPE=91.62mm (tiny fast config unreliable); GPU free |
| v30 smoke val1 | v30 hierarchical + physical loss, val_stride=1 | RTX 4090 | Queued as first job in v31 top-5 smoke queue |
| v31 top-5 smokes | domain_balanced, physical_floor_only, hierarchical_more_dropout, outlier_adaptive, epipolar_guided | RTX 4090 | Running sequentially via scripts/run_v31_top5_local4090_queue.sh |
| v31 geometry attention | geometry-biased hierarchical encoder wired into v5 model | RTX 4090 | Ready to smoke after top-5 queue finishes |
| v31 camera embedding | geometry-aware pairwise camera view embedding wired into v5 model | RTX 4090 | Queued in second-wave smoke queue after top-5 finishes |
| v31 second wave | geometry attention + camera embedding + physical collision + skeleton residual gate smokes | RTX 4090 | Waiting in scripts/run_v31_second_wave_local4090_queue.sh |
| v32 first wave | domain-aware + TCR + ray attention smokes | RTX 4090 | Queued after v31 top-5 via scripts/run_v32_first_wave_local4090_queue.sh |
| v31 top-5 A800 | domain_balanced, physical_floor, hierarchical_more_dropout, outlier, geometry_attention | A800-D | Running on GPUs 4-7; blocking v32/v33 A800 queue until they finish |
| v32 next | Domain-aware view curriculum, TCR, ray attention, outlier triangulation, bounded physical | merged | Domain-aware + TCR + ray attention + bounded physical alignment merged |
| v33 uncertainty-aware triangulation | Per-view log-variance + precision-weighted DLT | merged | Smoke 82.02mm; full local run in progress |
| v33 outlier-view rejection | Learned feature-aware outlier detector | merged | Smoke + wired into OmniMultiViewFusionV5 |
| v33 ray-conditioned attention | Geometry-biased ray cross-view attention | merged | Smoke + wired into OmniMultiViewFusionV5 |
| v33 combined | uncertainty + outlier + ray | local 4090 | Stopped: epoch 2 val_MPJPE=142.98mm (overfit); fixed run restarted with weight_decay and lower outlier supervised weight |
| v33 combined fixed | same + weight_decay 1e-4 + outlier_supervised_weight 0.01 | local 4090 | Full run reached epoch 1 val 29.67mm then stopped to free GPU; A800 queued |
| v33 hierarchical multi-scale spatial pyramid | per-scale geometry-biased cross-view attention | merged | Quick smoke 55.59mm; full local run reached epoch 1 val_MPJPE=27.32mm; A800 queued |
| v34 view-joint graph network | (view, joint) graph attention over bone/symmetry/cross-view edges | merged | Quick smoke 54.07mm vs v31 55.28mm; restarted full run reached epoch 1 val_MPJPE=27.17mm (was 27.60mm); A800 queued |
| v34 geometry-aware view-joint graph network | v34 VJGN with epipolar + ray-intersection bias on cross-view edges | merged | Full local RTX 4090 run started; A800 queued (2-layer, 1-layer, dropout, HMSP stack, combined-max); smoke 82.05mm |
| v34 local ablations queue | HMSP+geometry VJGN and v33 combined-fixed+geometry VJGN | RTX 4090 | v33 HMSP, v34 VJGN, and v34 geometry-aware VJGN full runs stopped to free GPU for quick ablation; can restart later |
| v34 quick ablations | VJGN vs geometry-aware VJGN (20 samples/seq, 5 epochs) | RTX 4090 | Done: v34 VJGN best 71.32mm; geometry-aware VJGN best 97.28mm; plain VJGN selected for v35 base |
| v34 HMSP + VJGN stack | v33 HMSP + v34 VJGN together | A800-D | Queued in v33/v34 A800 queue; quick smoke 55.96mm |
| v34 HMSP + geometry-aware VJGN stack | v33 HMSP + v34 geometry-aware VJGN together | local 4090 | Full local run reached epoch 1 val_MPJPE=25.50mm; A800 queued |
| v35 temporal view-joint graph network | v34 VJGN + temporal edges across frames | RTX 4090 | Full local run reached epoch 1 val_MPJPE=27.08mm (vs v34 VJGN 27.17mm); temporal edges give small gain; still running epoch 2 |
| v35 A800 queue | v35 TVJGN on top of v34 VJGN / geometry-aware VJGN | A800-D | Added to launch_v33_a800_queue.py; poller restarted; base v34 VJGN selected after quick ablation |
| v36 uncertainty-gated iterative graph refinement | v35 TVJGN + per-node uncertainty gating + iterative refinement; self-evolving fusion | merged | Smoke passed (clip_len=3, d=32, 10 samples): val_MPJPE 100.47mm; full local epoch 1 val_MPJPE=26.42mm (vs v35 27.08mm); stopped after epoch 2 overfit to 76.88mm; A800 queue running |
| v37 self-critique view reliability | v36 UGIGR + per-(view,joint) reliability score learned from reprojection residuals; self-evolution | merged | Smoke 97.24mm; full local epoch 1 val_MPJPE=26.94mm (worse than v36 26.42mm); A800 queue entry added |
| v38 expanded WebBridge data | v37 SCVR + expanded H36M/MPI manifest (104 train / 16 val files) | ready | Manifest and A800 queue entry created; local run script added |
| v39 reliability-coupled adaptive graph refinement | v38 + v37 reliability gates v36 uncertainty; closes the self-evolution loop | RTX 4090 | Implemented; smoke passed (82.03 mm tiny smoke); A800 queue entry added |
| v40 skeleton-aware physical loss | v39 + composite bone/joint/symmetry/floor physical prior | RTX 4090 | Implemented; smoke passed (82.03 mm tiny smoke); A800 queue entry added |
| v41 weighted domain loss | v40 + per-domain MSE weights for WebBridge mixed training | RTX 4090 | Implemented; smoke passed (82.03 mm tiny smoke); A800 queue entry added |
| v42 v36+physical+domain (no v37) | v36 + v40 physical loss + v41 domain weights; tests whether gains require v37 | RTX 4090 | Bug #151 fixed; running via `scripts/nohup_v42_v43_queue_local_4090.sh`; A800 queue entry added |
| v43 adaptive per-node residual | v42 + scale v36 UGIGR residual by per-node uncertainty gate | RTX 4090 | Smoke passed (tiny, 10 samples): epoch 2 val_MPJPE 100.53 mm (not representative); full run waiting for GPU; A800 queue entries: base v43, scaled d128/10k-samples, full WebBridge all-train mixed (1333/156 files), and v25 all-train baseline; issue #152 |
| v44 edge-type-aware uncertainty gating | v43 + learned per-edge-type temperature for the v36 source gate | RTX 4090 | Smoke passed (tiny, 10 samples): epoch 2 val_MPJPE 100.51 mm (not representative); waiting for v43 results before A800 queue; issue #153 |
| v33 HMSP A800 | full scale after v31 top-5 | A800-D | Queued in v33 A800 queue |
| v32/v33/v34 A800 queue | v31_physical_floor_only, v32 x5, v33 x4, v34 x2, HMSP stacks | A800-D | Poller fixed to include variant prefix in tmux session matching; co-locates when GPU memory >= 30 GiB; v31/v32/v33 runs launching on GPUs 4-7; v34/v35/v36 entries queued |
| GitHub issues/PRs | Use API token from git remote URL | active | Issues/PRs created and merged via curl/GitHub API |

### v29 SEH-MV A800-D runs

| Run | Description | GPU | Status |
|-----|-------------|-----|--------|
| v29 full SEH-MV | v25 + v29 hierarchical + TTE + physical loss (batch 24, d=128) | — | Killed; TTE at inference produces ~90mm val regardless of scale |
| v29a | hierarchical encoder only (no TTE, no physical) | GPU1 | Epoch 1 val_MPJPE=28.12mm; epoch 2=47.85mm; epoch 3=81.08mm (severe overfitting) |
| v29b | hierarchical + TTE (no physical) | GPU2 | Killed; epoch 1 val_MPJPE=90.35mm — TTE implementation is broken |
| v29d | TTE + physical (no hierarchical) | GPU3 | Killed; epoch 1 val_MPJPE=90.28mm — same TTE failure |
| v29 20-run sweep | Hierarchical-only + physical-loss ablations (TTE disabled) | GPU4/5/6/7 | Stopped to free GPUs for v31; recorded vals: v29o 21.54, v29u 27.58, v29z 28.02, v29a 81.08, v29b/d ~90 mm |
| v30a | v30 hardened hierarchical + physical loss (d=128, full) | — | Poller waiting for >65 GiB free GPU on A800-D; previous attempt OOM’d on GPU0 due to VLLM worker |
| v30 smoke | v30 hardened hierarchical + physical loss (d=64, local smoke) | RTX 4090 | Epoch 1 val_MPJPE=94.15mm (fast 50-sample smoke; not representative) |

- **Remote host:** `a800-D` (SSH)
- **Remote repo:** `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`
- **Access rule:** A800-D code is synced from the local repo via `git archive` + `scp` because GitHub HTTPS access from A800 is intermittent. The poller (`scripts/launch_v33_a800_queue.py`) archives the local tracked files and extracts them on A800 before launching runs.
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
