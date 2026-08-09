# v49: Lightweight Architecture for RTX 4090

**Status:** Proposal  
**Labels:** `experiment`, `P1-next`, `infra`  
**Tracking issue:** #166 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain)  

---

## 1. Problem Statement

The v46–v48 stack (sparse-view generalization → temporal aggregation → domain generalization) is increasingly heavy:

- v47 adds a 2-layer temporal transformer on top of v46.
- v48 adds domain-conditional FiLM/GRL/DDWL on top of v47.
- Full runs require A800-D with `d=128`, `batch_size=16`, `clip_len=13`, and long queues.
- Local RTX 4090 smoke tests are now the **only** fast feedback loop, but they are ad-hoc: each variant borrows a different subset of flags, and there is no stable "4090 baseline" that the team can train to convergence locally.

This creates three concrete problems:

1. **Iteration speed.** A800 priority runs can block for days; local 4090 experiments cannot currently reproduce the full v46–v48 pipeline end-to-end.
2. **Reproducibility.** New contributors lack a deterministic, affordable config that matches the paper pipeline without A800 access.
3. **Architecture bloat.** Each new variant layers modules on top of the previous one; no design effort has been spent on a *minimal* architecture that preserves the paper claims (multi-view, sparse-view, temporal, domain) at 4090 scale.

v49 proposes a single **lightweight architecture** that implements the same paper story as v46–v48 but is small enough to train and validate entirely on an RTX 4090.

---

## 2. Proposed Approach

Build a compact variant of the v47/v48 stack by replacing its heavy components with cheaper, functionally equivalent counterparts:

1. **Replace the v47 temporal transformer with a lightweight causal temporal MLP.** Instead of a 2-layer transformer with `d_model=64`, use a small causal Conv1D over time with depthwise separable convolutions. This preserves temporal smoothing but cuts memory and runtime.
2. **Replace the v48 domain adapter with domain-conditional batch norm + DDWL only.** Drop the FiLM/GRL module; keep the v41 dynamic domain weighting loss, which is cheap and already implemented.
3. **Keep v46 sparse-view generalization as-is**, but reduce `v46_svg_hidden` from 64 to 32 and use a single-layer reliability MLP.
4. **Freeze the v25 geometry-fusion backbone** for the first epoch to stabilize the small local run, then unfreeze.
5. **Self-evolution feedback loop:** reuse the v37 self-critique reliability head as a lightweight per-view uncertainty estimator. The reliability score is fed back into the v46 reliability weights, creating a closed uncertainty→triangulation→reliability loop without extra heavy modules.

The resulting model is:

```text
multi-view video
    |
    ▼
v25 MultiViewGeometryFusionV5 backbone (frozen epoch 0)
    |
    ▼
v46 Sparse-View Generalization (lightweight: hidden=32, 1-layer MLP)
    |
    ▼
v37 Self-Critique View Reliability (lightweight per-view uncertainty)
    │
    └─> feedback loop updates v46 reliability weights
    ▼
v47-Lite Temporal Aggregation (causal depthwise Conv1D, no transformer)
    ▼
v48-Lite Domain Adaptation (domain-conditional BN + v41 DDWL only)
    ▼
3-D pose
```

---

## 3. Concrete Code-Level Changes

### New files

- `motionflow_mv/fusion/temporal_aggregation_v49_lite.py`
  - `TemporalAggregationV49Lite`: causal depthwise separable Conv1D over `(B, T, J, 3)`.
  - Inputs: `poses_3d (B, T, J, 3)`, `view_mask (B, T, V)`, optional `clip_mask (B, T)`.
  - Output: refined `poses_3d (B, T, J, 3)` with a residual gate initialized to `0.0`.
- `motionflow_mv/fusion/domain_adapter_v49_lite.py`
  - `DomainAdapterV49Lite`: domain-conditional batch norm only; no FiLM/GRL.
- `configs/benchmark_v49_lite_4090_smoke.yaml`
  - 4090 smoke config: `d=64`, `batch_size=4`, `clip_len=9`, `train_samples=500`, 5 epochs.
- `configs/benchmark_v49_lite_4090_full.yaml`
  - 4090 full config: `d=64`, `batch_size=4`, `clip_len=13`, `train_samples=4000`, 10 epochs.
- `scripts/run_v49_lite_4090_smoke.sh`
  - Smoke launch script for local RTX 4090.

### Modified files

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add flags:
    - `use_v49_lite_architecture` (default `False`)
    - `use_v49_lite_temporal` (default `False`)
    - `use_v49_lite_domain` (default `False`)
  - Wire `TemporalAggregationV49Lite` and `DomainAdapterV49Lite` when flags are set.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Expose CLI flags for v49-lite.
  - Add `v49_lite_freeze_backbone_epochs` (default `1`).
  - Use v37 self-critique head in lightweight mode when `use_v49_lite_architecture=true`.
- `experiments/eval_variable_views.py`
  - Report `MPJPE@k` for `k = 2, 3, 4, full` and wall-clock latency (ms/frame).

### New training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_lite_architecture` | bool | `False` | Master switch for the 4090-lightweight stack. |
| `use_v49_lite_temporal` | bool | `False` | Enable causal depthwise Conv1D temporal head. |
| `use_v49_lite_domain` | bool | `False` | Enable domain-conditional BN + DDWL. |
| `v49_lite_temporal_hidden` | int | `32` | Hidden channels for the temporal Conv1D. |
| `v49_lite_temporal_kernel` | int | `5` | Temporal kernel size; pad to preserve length. |
| `v49_lite_freeze_backbone_epochs` | int | `1` | Freeze v25/v46 backbone for the first N epochs. |
| `v49_lite_use_v37_feedback` | bool | `True` | Use v37 self-critique reliability as feedback into v46 weights. |

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|------|------------|
| Causal Conv1D under-matches the v47 transformer at sparse views. | Keep the transformer as an optional flag; compare in ablation. |
| Removing v48 FiLM/GRL hurts 3DPW actual-mode transfer. | Rely on v41 DDWL and domain-conditional BN; evaluate cross-domain gap. |
| v37 feedback loop destabilizes the lightweight stack. | Gate the feedback weight and initialize to `0.0`; ramp over warmup. |
| 4090 still OOM at `clip_len=13`. | Default smoke to `clip_len=9`; full run uses gradient accumulation if needed. |
| Local run overfits quickly on small data. | Keep dropout, stochastic depth, and early stopping from v30/v31. |

---

## 5. Success Metrics and Experiments

### Smoke experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_lite_4090_smoke.yaml` |
| Hardware | Local RTX 4090 (24 GiB) |
| Duration | ~1–2 hours |
| Goal | `val_MPJPE < 80 mm`, no NaN/OOM, latency < 50 ms/frame |
| Expected | ~70–75 mm, comparable to v46 smoke baseline |

### Full experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_lite_4090_full.yaml` |
| Hardware | Local RTX 4090 |
| Duration | ~8–12 hours |
| Goal | `val_MPJPE < 35 mm` at full views; `MPJPE@2` within 15% of `MPJPE@full` |
| Expected | ~30–32 mm, proving the full v46–v48 pipeline can run on a single consumer GPU |

### Evaluation

Run:

```bash
python experiments/eval_variable_views.py \
  --checkpoint outputs/v49_lite_4090_full/best.pth \
  --config configs/benchmark_v49_lite_4090_full.yaml \
  --view_subsets 2,3,4,full \
  --out outputs/v49_lite_eval.json
```

Key metrics:

- `val_MPJPE@full`
- `MPJPE@2`, `MPJPE@3`, `MPJPE@4`
- `latency_ms_per_frame`
- `domain_gap` (if v48-lite domain adapter enabled)

---

## 6. Self-Evolution Feedback Loop

v49 keeps the self-evolution loop introduced in v36/v37, but in a lightweight form:

1. v37 self-critique predicts a per-view reliability score from reprojection residuals.
2. This score is used to re-weight the v46 sparse-view reliability head.
3. The weighted v46 output is triangulated, refined by v49-lite temporal smoothing, and supervised by the 3-D ground truth.
4. The reprojection residual of the refined pose is computed and fed back to update the v37 reliability estimator.

This loop is closed within a single forward/backward pass, so it adds no extra training stage and fits the 4090 memory budget.

---

## 7. Paper Story Fit

v49 supports the paper claim: *Our multi-view pose pipeline is practical and accessible: the same geometric-fusion, sparse-view, temporal, and domain-generalization ideas can be trained to convergence on a single consumer GPU without sacrificing the core paper claims.* It also provides a reproducible baseline for reviewers and contributors who do not have A800 access.

---

## 8. Next Steps

1. Implement `TemporalAggregationV49Lite` and `DomainAdapterV49Lite`.
2. Add v49-lite flags to `OmniMultiViewFusionV5` and the trainer.
3. Run smoke on RTX 4090 and compare against `configs/benchmark_v46_svg_smoke.yaml`.
4. If smoke passes, run the full 4090 config.
5. Document the final 4090 baseline in `AGENTS.md` and update the local run status table.
