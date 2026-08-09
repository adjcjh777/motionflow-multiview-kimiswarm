# v50 Adaptive Temporal Window (ATW)

## One-sentence idea

Replace the fixed-length temporal context of v47/v49-Lite temporal aggregation with a **causal, per-joint adaptive temporal window** that learns how much past motion history each joint needs, so the model uses short windows for fast/ambiguous motion and longer windows for slow/certain motion.

## Architecture description

`AdaptiveTemporalWindowV50` sits between the v46 sparse-view fusion output and the final 3-D pose head. It keeps two parallel temporal encoders: a **short-window branch** (causal, receptive field `W_short`) and a **long-window branch** (causal, receptive field `W_long`). A lightweight motion-aware gating network consumes per-joint temporal differences (velocity) and the v46 reliability features, and outputs a soft scalar `α_t,j ∈ [0,1]` per joint `j` and time `t`. The final per-joint temporal token is the convex combination `α · long_branch + (1 − α) · short_branch`. Both branches share the same causal Conv1D/Transformer block from v49-Lite so the only new parameters are the gating MLP and the duplicated first layer of the long branch. The module is identity-at-init by setting `α = 0.5` everywhere, and it is fully causal, so it can run in a streaming evaluation.

## New config flags

| Flag | Default | Description |
|---|---|---|
| `use_v50_adaptive_temporal_window` | `False` | Master switch. |
| `v50_atw_short_window` | `3` | Short causal window (frames). |
| `v50_atw_long_window` | `9` | Long causal window (frames). |
| `v50_atw_d_model` | `64` | Feature dimension (matches v49-Lite). |
| `v50_atw_gating_hidden` | `32` | Hidden size of the joint-level gating MLP. |
| `v50_atw_motion_feature` | `"velocity"` | Motion cue: `"velocity"` (preferred) or `"residual"`. |
| `v50_atw_alpha_init` | `0.5` | Initial gating value; set to 0.5 for identity init. |

## Loss term

A small **temporal-window regularization loss** keeps the gate from collapsing:

`L_atw = λ · E[ α · (1 − p_long) + (1 − α) · p_long ]`

where `p_long` is the empirical fraction of joints that *should* use the long window (we set `p_long = 0.3` from validation motion statistics). Default `v50_atw_loss_weight = 0.01`. This encourages the model to actually use both branches while the main pose loss drives the selection.

## Evaluation metric

Primary metrics are `val_MPJPE@full`, `MPJPE@2`, `MPJPE@3`, and `MPJPE@4` from the canonical v49 protocol. We add a **temporal stability** auxiliary metric: per-sequence `MPJPE_t − MPJPE_{t+1}` standard deviation (`Jitter@full`). A successful run shows lower MPJPE *and* lower Jitter without regressing full-view accuracy.

## Expected MPJPE impact

Based on the v46-SVG smoke result of 32.97 mm, the v49-Lite temporal head is expected to add a small gain of ~1–2 mm. AdaptiveTemporalWindowV50 should further improve `MPJPE@2/3/4` by **−2 mm to −3 mm** on sparse views and **−0.7 mm to −1.5 mm** on full views, with `Jitter@full` dropping by ~10 %. The largest gains are anticipated on sequences with mixed fast/slow motion, where a fixed window currently under-samples fast actions or over-smooths slow ones.

## Main risk / mitigations

| Risk | Mitigation |
|---|---|
| **Gating collapses** to all-short or all-long. | Initialize `α = 0.5`; use the regularization loss above; freeze short branch for the first 500 steps. |
| **Causal streaming latency** increases if `W_long` is too large. | Cap `v50_atw_long_window ≤ 13` and evaluate frame-rate on 4090 smoke. |
| **Overfitting on small smoke data** (500 samples). | Keep gating MLP tiny (one hidden layer); tie weights with v49-Lite temporal branch where possible. |
| **v49-Lite not yet proven.** | Gate v50 experiments on passing v49-Lite smoke; do not start until v49-Lite `val_MPJPE` is finite and stable. |

## Dependencies and sequencing

- Depends on: v46 sparse-view generalization, v47 temporal aggregation, and v49-Lite temporal aggregation.
- Smoke config: `configs/benchmark_v50_adaptive_temporal_window_smoke.yaml`
- Smoke goal: `val_MPJPE@full` within 1 mm of v49-Lite, `MPJPE@2/3/4` improved by ≥2 %, no NaN/OOM, `Jitter@full` ≤ v49-Lite.
