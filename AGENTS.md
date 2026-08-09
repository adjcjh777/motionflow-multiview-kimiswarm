# MotionFlow-MultiView Agent Notes

This file captures the current A800-D workflow, tmux conventions, and issue labels used in the ICRA/CVPR 2027 multi-view pose project.

## A800-D status snapshot

| Run | Description | GPU | Status |
|-----|-------------|-----|--------|
| v25 geometry fusion full | v18 + geometry fusion (full WebBridge/H36M/MPI) | GPU4 (historical) | **Best val_MPJPE 17.17 mm**; still the strongest known baseline |
| v25 all-train baseline | v25 on h36m+mpi mixed manifest (A800 lacks aistpp/3dpw) | GPU6 | **Running** — launched after GPU6 freed ~35 GiB |
| v25 + physical + domain | v25 + v40 skeleton-aware physical loss + v41 domain weights | GPU6 | **Running** — launched after GPU6 freed ~35 GiB |
| v42 v36+physical+domain | v36 UGIGR + v40 physical loss + v41 domain weights (no v37) | GPU6 | **Running** — launched after GPU6 freed ~35 GiB; local epoch-1 26.16 mm (d=64) |
| v43 adaptive per-node residual base | v42 + residual scaled by per-node uncertainty gate | GPU6 | **Running** — launched after GPU6 freed ~35 GiB |
| v43 adaptive per-node residual scaled | v43 with d=128 / 10k samples | — | **Queued** |
| v43 adaptive per-node residual all-train | v43 on full WebBridge all-train mixed | — | **Queued** |
| v31 top-5 A800 | domain_balanced, physical_floor, hierarchical_more_dropout, outlier, geometry_attention | GPUs 4-7 | Completed / stopped to free GPUs for v33/v34 queue |
| v32/v33/v34/v35/v36/v37/v38/v39/v40/v41 queue | Complex-stack variants (TCR, ray attention, UGIGR, SCVR, etc.) | — | Queued behind the v25/v42/v43 priority runs |
| v44 edge-type-aware uncertainty gating | v43 + learned per-edge-type temperature for the v36 source gate | — | Smoke passed; full run on hold pending A800 v25/v42/v43 results |
| v45-AGF | v25 + adaptive per-view/per-joint triangulation weights | RTX 4090 | Fast smoke done (1 epoch, 50 samples); medium run (500 samples, 5 epochs) now running; full A800 run queued |
| v46-SVG | v25/v45 + sparse-view generalization (view dropout + reliability head) | A800-D | **Queued** after v45-AGF results; see `docs/swarm_iter23_action_plan.md` (#160) |
| v47-temporal | v46-SVG + temporal aggregation across (time, joint) tokens | A800-D | **Queued** after v46-SVG smoke; see `docs/swarm_iter24_action_plan.md` (#162) |
| v48-domain | v47 + domain-invariant sparse-view refinement + 3DPW actual-mode integration | A800-D | **Ready**; blocked until v47 smoke completes; see `docs/swarm_iter25_action_plan.md` (#164) |

## v46/v47/v48/v49 A800 runs

| Run | Description | GPU | Status |
|-----|-------------|-----|--------|
| v46-SVG full | v45-AGF + sparse-view generalization (`p=0.3`, `min_views=2`) + reliability head | GPU6 | **Running** since 2026-08-09 06:19; first validation pending |
| v47-temporal full | v46-SVG + temporal aggregation across (time, joint) tokens | A800-D | **Done**; best val_MPJPE 63.37 mm |
| v48-domain full | v47 + domain-conditional FiLM/GRL/DDWL + 3DPW actual-mode integration | A800-D | **Running** on GPU6 |
| v51-CDSVR full | v50 SEFH + domain-conditioned sparse-view reliability/uncertainty | A800-D | **Queued**; see issue #181 |
| v49-Lite full | v46-SVG + lightweight causal temporal aggregation (v47 replacement) | A800-D | **Queued** after v46/v47 results; see issue #175 |
| v50 SEFH full | v46-SVG + Self-Evolution Feedback Head | A800-D | **Queued** after v49-Lite results; see issue #176 |
| v51 DAE on v46 | v46-SVG + Domain-Agnostic Ensemble of pose experts | A800-D | **Queued** in `launch_v33_a800_queue.py`; see issue #178 |
| v51 DAE on v50 | v50 SEFH + Domain-Agnostic Ensemble of pose experts | A800-D | **Queued** after v50 SEFH results; see issue #178 |
| v52 UWT on v50/v51 | v45/v46/v50/v51 + learnable uncertainty-weighted DLT triangulation | A800-D | **Queued** in `launch_v33_a800_queue.py`; see issue #182 |
| v53 PSC on v52 | v45/v46/v50/v51/v52 + physical-space calibration (floor + bone + gated residual) | A800-D | **Smoke promising** (tiny 102.56 mm; local medium v2 epoch-1 66.06 mm, running); A800 full run queued with `v53_psc_loss_weight=0.1`, warmup and weight-decay; see issue #183 |
| v54 PSC-v2 on v53 | v45/v46/v50/v51/v52/v53 + skeleton-graph joint-level physical refiner | A800-D | **Dropped** after smoke showed no gain; full run removed from queue; see issue #184 |
| v55 ORR on v54 | v53/v54 + outlier-robust Cauchy reliability head | A800-D | **Dropped** after smoke showed no gain / instability; full run removed from queue; see issue #185 |
| v52 scale v45/v46 | v45-AGF + v46-SVG scaled (d=128, 10k samples, 10 epochs) | A800-D | **Queued** in `launch_v33_a800_queue.py`; see issue #179 |

## Local RTX 4090 status snapshot

### Current local runs (2026-08-09)

| Run | Description | Status |
|-----|-------------|--------|
| v46-SVG smoke | v25/v45 + on-the-fly view dropout (`p=0.3`, `min_views=2`) + reliability head | Stopped to free GPU; epoch-1 val_MPJPE 30.57 mm |
| v47 temporal aggregation smoke | v46-SVG + lightweight temporal refinement head (`d_model=64`, `num_layers=2`) | Done; best val_MPJPE 63.37 mm on A800 |
| v48 domain generalization smoke | v47 + domain-conditional FiLM/GRL/DDWL + 3DPW actual val | Launched on A800 (GPU6) |
| v51 CDSVR smoke | v50 SEFH + domain-conditioned cross-attention reliability/uncertainty | Running on RTX 4090; tracking issue #181 |
| v49-Lite smoke | v46-SVG + lightweight causal temporal aggregation (v47 replacement) | Queued in A800 queue (issue #175) |
| v49 ablation matrix | v45/v46/v47/v48/v49-Lite 2-epoch comparison | Ready; waiting for GPU |
| v50 design swarm | 20-agent design of next module | Done; top-1 = Self-Evolution Feedback Head (issue #176) |
| v50 SEFH smoke | v46-SVG + Self-Evolution Feedback Head | Ready; smoke script added; run after v46/v47/v48 chain |
| v51 DAE smoke | v46-SVG + Domain-Agnostic Ensemble of pose experts | **Done**; best val_MPJPE 35.29 mm (vs v46 32.97 mm); needs tuning; see issue #178 |
| v52 UWT smoke | v45/v46/v50/v51 + learnable uncertainty-weighted DLT triangulation | **Done**; tiny smoke best 78.68 mm; medium best 60.09 mm; see issue #182 |
| v53 PSC smoke | v45/v46/v50/v51/v52 + physical-space calibration (floor + bone + gated residual) | **Fixed NaN root cause** (`sqrt(weights).clamp` -> `clamp().sqrt()`); also clamped canonical `log_bone_lengths` in v53 to prevent unbounded exponential growth; A800 tiny smoke best 98.65 mm; local tiny smoke best 78.76 mm; v2 medium diverged (train loss >11); v3 medium (same recipe + clamp fix) now running; see issue #183 |
| v54 PSC-v2 smoke | v45/v46/v50/v51/v52/v53 + skeleton-graph joint-level physical refiner | **Dropped**; A800/Local tiny smoke ~103 mm (worse than v52 78.68 mm); not beneficial; see issue #184 |
| v55 ORR smoke | v53/v54 + outlier-robust Cauchy reliability head before UWT | **Dropped**; local tiny smoke best 98.11 mm, A800 epoch-2 loss exploded; not beneficial; see issue #185 |
| v56 APL smoke | v53 + learned per-sample PSC loss weight from uncertainty/pose std | **Dropped**; local tiny smoke best 104.30 mm (worse than v53 78.76 mm); not beneficial |

### Historical / detailed local run log

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
| v34 HMSP + geometry-aware VJGN stack | v33 HMSP + v34 geometry-aware VJGN together | local 4090 | Process was left running and sharing 4090 with v42; killed to free GPU; epoch 1 val_MPJPE=25.50mm recorded; A800 queued |
| v35 temporal view-joint graph network | v34 VJGN + temporal edges across frames | RTX 4090 | Full local run reached epoch 1 val_MPJPE=27.08mm (vs v34 VJGN 27.17mm); temporal edges give small gain; still running epoch 2 |
| v35 A800 queue | v35 TVJGN on top of v34 VJGN / geometry-aware VJGN | A800-D | Added to launch_v33_a800_queue.py; poller restarted; base v34 VJGN selected after quick ablation |
| v36 uncertainty-gated iterative graph refinement | v35 TVJGN + per-node uncertainty gating + iterative refinement; self-evolving fusion | merged | Smoke passed (clip_len=3, d=32, 10 samples): val_MPJPE 100.47mm; full local epoch 1 val_MPJPE=26.42mm (vs v35 27.08mm); stopped after epoch 2 overfit to 76.88mm; A800 queue running |
| v37 self-critique view reliability | v36 UGIGR + per-(view,joint) reliability score learned from reprojection residuals; self-evolution | merged | Smoke 97.24mm; full local epoch 1 val_MPJPE=26.94mm (worse than v36 26.42mm); A800 queue entry added |
| v38 expanded WebBridge data | v37 SCVR + expanded H36M/MPI manifest (104 train / 16 val files) | ready | Manifest and A800 queue entry created; local run script added; 3DPW loader support added in `webbridge_mixed_dataset.py` |
| v39 reliability-coupled adaptive graph refinement | v38 + v37 reliability gates v36 uncertainty; closes the self-evolution loop | RTX 4090 | Implemented; smoke passed (82.03 mm tiny smoke); A800 queue entry added |
| v40 skeleton-aware physical loss | v39 + composite bone/joint/symmetry/floor physical prior | RTX 4090 | Implemented; smoke passed (82.03 mm tiny smoke); A800 queue entry added |
| v41 weighted domain loss | v40 + per-domain MSE weights for WebBridge mixed training | RTX 4090 | Implemented; smoke passed (82.03 mm tiny smoke); A800 queue entry added |
| v42 v36+physical+domain (no v37) | v36 + v40 physical loss + v41 domain weights; tests whether gains require v37 | RTX 4090 | Epoch 1 val_MPJPE=26.16mm; process got stuck/dup after step 1700 and was killed; A800 queue entry added; snapshot in `docs/results_snapshot_2026_08_09.md` |
| v25 + physical + domain | v25 geometry fusion + v40 physical loss + v41 domain weights | RTX 4090 | Epoch 1 val_MPJPE=27.71mm (d=64); worse than v42 local 26.16mm; A800 d=128 run still pending |
| v43 adaptive per-node residual | v42 + scale v36 UGIGR residual by per-node uncertainty gate | RTX 4090 | Local full run at step ~3600 (loss 6.38, no epoch-1 val yet); A800 queue entries: base v43, scaled d128/10k-samples, full WebBridge all-train mixed (1333/156 files), and v25 all-train baseline; issue #152 |
| v44 edge-type-aware uncertainty gating / v44 decision plan | v43 + learned per-edge-type temperature for the v36 source gate; v44 architecture chosen conditionally | RTX 4090 | Smoke passed; `docs/v44_decision_plan.md` drafted; final branch chosen after A800 results; issue #153 / #154 |
| A800 queue priority | v25 all-train baseline, v25+physical+domain, v42, v43 base/scaled/all-train moved to front of `launch_v33_a800_queue.py` | A800-D | Reordered and committed; poller restarted; waiting for GPU 4-7 to free |
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

## Next steps and v44 decision plan

Snapshot date: 2026-08-09.

The v44 architecture will be chosen conditionally once the priority A800 runs finish:

1. Wait for A800 results from the priority queue: `v25 all-train baseline`, `v25 + physical + domain`, `v42`, and `v43` variants.
2. Compare epoch-1 val_MPJPE against the v25 baseline (~17 mm on A800; local v42/v43 ~26 mm).
3. Choose a branch (see `docs/v44_decision_plan.md` for full details):
   - **Branch A (v25 wins / most likely as of 2026-08-09):** v44 = v25 + selective additions (physical loss + domain weights; then possibly outlier/variable-view augmentation).
   - **Branch B (v42 beats v25):** Keep the v31-v36 stack and add stronger regularization (higher weight decay, dropout, stochastic depth, SWA/EMA).
   - **Branch C (v43 beats v42):** Keep the adaptive per-node residual and explore edge-type-aware gating.
   - **Branch D (capacity/data wins):** Scale the winning architecture (d=128, n_st_layers=3, batch_size=16, clip_len=13, train_samples=10000).
4. Update `docs/results_snapshot_2026_08_09.md` with the A800 numbers.
5. Run v44 smoke on the local RTX 4090, then queue a full A800 run.

No new GPU training should start until the A800 priority queue results are in and the branch is chosen.

## v46 sparse-view generalization (SVG) conventions

Tracking issue: #160. Depends on: #154, v45-AGF.

- **Branch:** `v46-svg`
- **Module:** `motionflow_mv/fusion/sparse_view_generalization_v46.py`
- **Augmentation:** `motionflow_mv/data/view_dropout_augmentation_v46.py`
- **Model wiring:** `motionflow_mv/fusion/omniview_fusion_v5.py`
- **Trainer:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Smoke config:** `configs/benchmark_v46_svg_smoke.yaml`
- **Smoke script:** `scripts/run_v46_svg_smoke_local_4090.sh`
- **Evaluation:** `experiments/eval_variable_views.py` reports `MPJPE@k` for `k = 2, 3, 4` plus full views.
- **A800 queue:** `scripts/launch_v33_a800_queue.py`
- **Flags:**
  - `use_v46_sparse_view_generalization`
  - `v46_svg_view_dropout_prob` (default 0.3)
  - `v46_svg_min_views` (default 2)
  - `v46_svg_hidden` (default 64)
  - `v46_svg_use_curriculum` (default True)
- **Workflow:**
  1. Do not start v46 smoke until the v45-AGF medium run frees the RTX 4090.
  2. Run `bash scripts/run_v46_svg_smoke_local_4090.sh`.
  3. If smoke val_MPJPE < 80 mm, add/confirm the A800 queue entry and start the full run.
  4. Update the status tables above with epoch-1 val and `MPJPE@k` numbers.
- **Labels:** `experiment`, `P1-next`

## v47 temporal aggregation conventions

Tracking issue: #162. Depends on: #160, v45-AGF.

- **Branch:** `v47-temporal`
- **Module:** `motionflow_mv/fusion/temporal_aggregation_v47.py`
- **Base:** reuses `SparseViewGeneralizationV46` reliability weights and view-dropout augmentation
- **Model wiring:** `motionflow_mv/fusion/omniview_fusion_v5.py`
- **Trainer:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Smoke config:** `configs/benchmark_v47_temporal_svg_smoke.yaml`
- **Smoke script:** `scripts/run_v47_temporal_svg_smoke_local_4090.sh`
- **Evaluation:** `experiments/eval_variable_views.py` reports `MPJPE@k` for v46 vs v47.
- **A800 queue:** `scripts/launch_v33_a800_queue.py`
- **Flags:**
  - `use_v47_temporal_aggregation`
  - `v47_temporal_d_model` (default `64`)
  - `v47_temporal_n_heads` (default `4`)
  - `v47_temporal_num_layers` (default `2`)
  - `v47_temporal_window` (default `None`)
  - `v47_temporal_dropout` (default `0.1`)
  - `v47_temporal_loss_weight` (default `0.01`)
  - `v47_use_view_count_conditioning` (default `True`)
- **Workflow:**
  1. Do not start v47 smoke until the v46-SVG smoke finishes.
  2. Run `bash scripts/run_v47_temporal_svg_smoke_local_4090.sh`.
  3. If smoke val_MPJPE < 75 mm, add/confirm the A800 queue entry and start the full run.
  4. Update the status tables above with epoch-1 val and `MPJPE@k` numbers.
- **Labels:** `experiment`, `P1-next`

## v48 domain generalization conventions

Tracking issue: #164. Depends on: #160, #162.

- **Branch:** `v48-domain`
- **Module:** `motionflow_mv/fusion/domain_adapter_v48.py`
- **Base:** reuses v46 Sparse-View Generalization and v47 Temporal Aggregation
- **Model wiring:** `motionflow_mv/fusion/omniview_fusion_v5.py`
- **Trainer:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Smoke config:** `configs/benchmark_v48_domain_smoke.yaml`
- **Smoke script:** `scripts/run_v48_domain_smoke_local_4090.sh`
- **Evaluation:** `experiments/eval_variable_views.py` reports per-domain `MPJPE@k` and 3DPW actual `MPJPE@1`.
- **A800 queue:** `scripts/launch_v33_a800_queue.py`
- **Flags:**
  - `use_v48_domain_generalization`
  - `v48_dg_hidden` (default `64`)
  - `v48_dg_grl_lambda` (default `0.1`)
  - `v48_dg_use_domain_film` (default `True`)
  - `v48_dg_use_ddwl` (default `True`)
  - `v48_dg_ddwl_temperature` (default `2.0`)
  - `v48_dg_ddwl_warmup_epochs` (default `1`)
  - `v48_3dpw_actual_val_paths` (default `None`)
  - `v48_dropout_per_domain` (default `{"0": 0.30, "1": 0.30, "5": 0.15}`)
- **Workflow:**
  1. Wait for v47-temporal smoke results.
  2. Run `bash scripts/run_v48_domain_smoke_local_4090.sh`.
  3. If smoke val_MPJPE is finite and per-domain metrics show no NaN/OOM, add/confirm the A800 queue entry and start the full run.
  4. Update the status tables above with epoch-1 val, per-domain `MPJPE@k`, and 3DPW actual `MPJPE@1` numbers.
- **Labels:** `experiment`, `P1-next`

## v52 uncertainty-weighted triangulation conventions

Tracking issue: #182. Depends on: #160, #162, #175, #176, #178, #181.

- **Branch:** `v52-uwt`
- **Module:** `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py`
- **Base:** reuses v45 Adaptive Geometry Fusion, v46 Sparse-View Generalization, v50 Self-Evolution Feedback Head, and v51 Cross-Domain Sparse-View Reliability
- **Model wiring:** `motionflow_mv/fusion/omniview_fusion_v5.py`, placed after the v25/v45 triangulation block
- **Trainer:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Smoke config:** `configs/benchmark_v52_uwt_smoke.yaml`
- **Smoke script:** `scripts/run_v52_uwt_smoke_local_4090.sh`
- **A800 queue:** `scripts/launch_v33_a800_queue.py` (entry `v52_uncertainty_weighted_triangulation_on_v50_v51`)
- **Flags:**
  - `use_v52_uncertainty_weighted_triangulation`
  - `v52_uwt_hidden` (default `64`)
  - `v52_uwt_n_layers` (default `2`)
  - `v52_uwt_weight_type` (default `"per_view_joint"`)
  - `v52_uwt_temperature` (default `1.0`)
  - `v52_uwt_use_geometry_bias` (default `True`)
  - `v52_uwt_use_feature_bias` (default `True`)
  - `v52_uwt_identity_init` (default `True`)
  - `v52_uwt_min_weight` (default `0.05`)
  - `v52_uwt_loss_weight` (default `0.01`)
  - `v52_uwt_damping` (default `1e-4`)
  - `v52_uwt_warmup_epochs` (default `0`)
- **Workflow:**
  1. Run `bash scripts/run_v52_uwt_smoke_local_4090.sh` on the local RTX 4090.
  2. Verify identity-at-init: loading a v51 checkpoint with v52 enabled should not change `val_MPJPE` by more than 0.1 mm.
  3. If smoke is stable (no NaN/Inf/OOM) and `val_MPJPE@full` is within 1 mm of the v51 baseline, start the A800 full run.
  4. Update the status tables above with epoch-1 val and `MPJPE@k` numbers.
- **Labels:** `experiment`, `P1-next`

## v53 physical-space calibration conventions

Tracking issue: #183. Depends on: #160, #162, #175, #176, #178, #181, #182.

- **Branch:** `v53-psc`
- **Module:** `motionflow_mv/fusion/physical_space_calibration_v53.py`
- **Base:** reuses v45 Adaptive Geometry Fusion, v46 Sparse-View Generalization, v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, and v52 Uncertainty-Weighted Triangulation
- **Model wiring:** `motionflow_mv/fusion/omniview_fusion_v5.py`, placed after the v52 UWT block and before the residual MLP
- **Trainer:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Smoke config:** `configs/benchmark_v53_physical_space_calibration_smoke.yaml`
- **Smoke script:** `scripts/run_v53_physical_space_calibration_smoke_local_4090.sh`
- **A800 queue:** `scripts/launch_v33_a800_queue.py` (entry `v53_physical_space_calibration_on_v52`)
- **Flags:**
  - `use_v53_physical_space_calibration`
  - `v53_psc_hidden` (default `64`)
  - `v53_psc_n_layers` (default `2`)
  - `v53_psc_identity_init` (default `True`)
  - `v53_psc_residual_gate_init` (default `-6.0`)
  - `v53_psc_use_uwt_weights` (default `True`)
  - `v53_psc_use_floor` (default `True`)
  - `v53_psc_use_bone_scale` (default `True`)
  - `v53_psc_loss_weight` (default `1.0`)
  - `v53_psc_floor_weight` (default `0.01`)
  - `v53_psc_bone_weight` (default `0.1`)
  - `v53_psc_reproj_weight` (default `0.1`)
  - `v53_psc_warmup_epochs` (default `0`)
  - `v53_psc_min_visible_views` (default `2`)
- **Workflow:**
  1. Run `bash scripts/run_v53_physical_space_calibration_smoke_local_4090.sh` on the local RTX 4090.
  2. Verify identity-at-init: loading a v52 checkpoint with v53 enabled should not change `val_MPJPE` by more than 0.1 mm.
  3. If smoke is stable (no NaN/Inf/OOM) and `val_MPJPE@full` is within 1 mm of the v52 baseline, start the A800 full run.
  4. Update the status tables above with epoch-1 val and `MPJPE@k` numbers.
- **Labels:** `experiment`, `P1-next`

## v57 domain-conditional physical-space calibration conventions

Tracking issue: #185. Depends on: #160, #162, #175, #176, #178, #181, #182, #183.

- **Branch:** `v57-dcpsc`
- **Module:** `motionflow_mv/fusion/domain_conditional_physical_calibration_v57.py`
- **Class:** `DomainConditionalPhysicalCalibrationV57`
- **Base:** reuses v45 Adaptive Geometry Fusion, v46 Sparse-View Generalization, v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, and v52 Uncertainty-Weighted Triangulation
- **Model wiring:** `motionflow_mv/fusion/omniview_fusion_v5.py`, placed after the v52 UWT block and before the final residual MLP / v47/v49 temporal / v50 SEFH heads
- **Trainer:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Smoke config:** `configs/benchmark_v57_domain_conditional_psc_smoke.yaml`
- **Smoke script:** `scripts/run_v57_domain_conditional_psc_smoke_local_4090.sh`
- **Evaluation:** `experiments/eval_variable_views.py` reports `MPJPE@k` for v57 vs v53 baselines.
- **A800 queue:** `scripts/launch_v33_a800_queue.py` (entry `v57_domain_conditional_psc_on_v52`)
- **Flags:**
  - `use_v57_domain_conditional_psc`
  - `v57_dcpsc_hidden` (default `64`)
  - `v57_dcpsc_n_layers` (default `2`)
  - `v57_dcpsc_num_domains` (default `8`)
  - `v57_dcpsc_use_floor` (default `True`)
  - `v57_dcpsc_use_bone_scale` (default `True`)
  - `v57_dcpsc_use_uwt_weights` (default `True`)
  - `v57_dcpsc_identity_init` (default `True`)
  - `v57_dcpsc_residual_gate_init` (default `-6.0`)
  - `v57_dcpsc_loss_weight` (default `1.0`)
  - `v57_dcpsc_floor_weight` (default `0.01`)
  - `v57_dcpsc_bone_weight` (default `0.1`)
  - `v57_dcpsc_reproj_weight` (default `0.1`)
  - `v57_dcpsc_warmup_epochs` (default `0`)
  - `v57_dcpsc_min_visible_views` (default `2`)
- **Workflow:**
  1. Run `bash scripts/run_v57_domain_conditional_psc_smoke_local_4090.sh` on the local RTX 4090.
  2. Verify identity-at-init: loading a v53 checkpoint with v57 enabled and `v57_dcpsc_loss_weight=0` should not change `val_MPJPE` by more than 0.1 mm.
  3. If smoke is stable (no NaN/Inf/OOM) and `val_MPJPE@full` is within 1 mm of the v53 baseline, start the A800 full run.
  4. Update the status tables above with epoch-1 val and `MPJPE@k` numbers.
- **Notes:**
  - v53 and v57 are mutually exclusive (raising `ValueError` if both are enabled) to avoid double PSC.
  - The domain-conditional heads use a shared `nn.Embedding(num_domains, hidden)` and FiLM (scale/shift) conditioning from the domain embedding.
  - Per-domain residual gate logits are sigmoided and broadcast per sample.
- **Labels:** `experiment`, `P1-next`

## v54 physical-space calibration v2 conventions

Tracking issue: #184. Depends on: #160, #162, #175, #176, #178, #181, #182, #183.

- **Branch:** `v54-psc-v2`
- **Module:** `motionflow_mv/fusion/physical_space_calibration_v2_v54.py`
- **Class:** `PhysicalSpaceCalibrationV2V54`
- **Base:** reuses v45 Adaptive Geometry Fusion, v46 Sparse-View Generalization, v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, v52 Uncertainty-Weighted Triangulation, and v53 Physical-Space Calibration
- **Model wiring:** `motionflow_mv/fusion/omniview_fusion_v5.py`, placed after the v53 PSC block and before the final residual MLP / v47/v49 temporal / v50 SEFH heads
- **Trainer:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Smoke config:** `configs/benchmark_v54_psc_v2_smoke.yaml`
- **Smoke script:** `scripts/run_v54_psc_v2_smoke_local_4090.sh`
- **A800 queue:** `scripts/launch_v33_a800_queue.py` (entry `v54_physical_space_calibration_v2_on_v53`)
- **Flags:**
  - `use_v54_physical_space_calibration_v2`
  - `v54_psc2_hidden` (default `64`)
  - `v54_psc2_n_layers` (default `2`)
  - `v54_psc2_num_domains` (default `8`)
  - `v54_psc2_use_floor` (default `True`)
  - `v54_psc2_use_contact` (default `True`)
  - `v54_psc2_use_bone_scale` (default `True`)
  - `v54_psc2_use_temporal_smoothness` (default `True`)
  - `v54_psc2_use_gnn` (default `True`)
  - `v54_psc2_gnn_layers` (default `1`)
  - `v54_psc2_identity_init` (default `True`)
  - `v54_psc2_residual_gate_init` (default `-6.0`)
  - `v54_psc2_loss_weight` (default `1.0`)
  - `v54_psc2_floor_weight` (default `0.01`)
  - `v54_psc2_bone_weight` (default `0.05`)
  - `v54_psc2_contact_weight` (default `0.01`)
  - `v54_psc2_temporal_weight` (default `0.01`)
  - `v54_psc2_reproj_weight` (default `0.1`)
  - `v54_psc2_contact_velocity_thresh` (default `0.3`)
  - `v54_psc2_min_visible_views` (default `2`)
  - `v54_psc2_warmup_epochs` (default `0`)
- **Workflow:**
  1. Run `bash scripts/run_v54_psc_v2_smoke_local_4090.sh` on the local RTX 4090.
  2. Verify identity-at-init: loading a v53 checkpoint with v54 enabled should not change `val_MPJPE` by more than 0.1 mm.
  3. If smoke is stable (no NaN/Inf/OOM) and `val_MPJPE@full` is within 1 mm of the v53 baseline, start the A800 full run.
  4. Update the status tables above with epoch-1 val and `MPJPE@k` numbers.
- **Labels:** `experiment`, `P1-next`

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

For the v52 -> v53 -> v54 medium smoke queue, use:

```bash
nohup bash scripts/nohup_v52_v53_v54_medium_queue_local_4090.sh > outputs/v52_v53_v54_medium_queue_nohup.log 2>&1 &
```

This waits for the running v52 UWT medium smoke to finish, then runs v53 PSC medium and v54 PSC-v2 medium sequentially.

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
