# v49: Integration and Next Experiment Queue

**Status:** Proposal / planning  
**Labels:** `experiment`, `P1-next`, `infra`  
**Tracking issue:** #167 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain), #154 (v25/v45 A800 results)

---

## 1. Problem Statement

The v46→v47→v48 chain is designed to be stacked, but its integration is currently implicit: each variant adds flags and modules without a unified recipe for *which combinations to run, in what order, and against which baselines*. This leads to three concrete problems:

1. **Unclear dependency ordering.** v46 must prove sparse-view robustness before v47 temporal aggregation can improve it; v47 must be stable before v48 domain generalization is meaningful. The existing `scripts/launch_v33_a800_queue.py` queues them in order, but there is no documented gating rule.
2. **No canonical ablation matrix.** We do not have a single queue entry that disables v46, v47, or v48 one at a time on the same base (v25 + v45-AGF), so we cannot isolate whether later modules help or just add capacity.
3. **Disconnect from the v25/v45 decision.** `docs/v44_decision_plan.md` showed that v25 geometry fusion is still the strongest baseline (~17.17 mm). v46-v48 are currently wired on top of v45-AGF, but the relative gain of the whole stack against a plain v25 baseline is unknown.

v49 therefore focuses on **integration hygiene and a disciplined next-experiment queue**: define the minimal runnable combinations, gate each stage with clear metrics, and feed the results back into a self-evolution loop.

---

## 2. Proposed Approach

### 2.1 Integration strategy

Build a **v48-full reference architecture** by stacking validated pieces on the v45-AGF base, but run it alongside a **v25-full baseline** so every addition is measured against the strongest known geometry-fusion model.

```text
Branch A (geometry-fusion family):
  v25 baseline  ->  v45-AGF  ->  v46-SVG  ->  v47-temporal  ->  v48-domain

Branch B (complex-stack family, legacy):
  v42/v43  ->  (optional) v46-SVG  ->  v47-temporal

Integration deliverables:
  1. A single YAML/config that turns on the full v46-v48 stack.
  2. Ablations that disable v46, v47, and v48 individually on the same base.
  3. A800 queue entries gated by smoke pass/fail and prior-stage val_MPJPE.
  4. Self-evolution feedback loop closed through v37 reliability + v45/v46 weights.
```

### 2.2 Where it fits in the pipeline

- **v25 MultiViewGeometryFusionV5** remains the triangulation backbone.
- **v45-AGF** supplies learnable per-(view,joint) triangulation weights; without it, v46 has no reliability signal to refine.
- **v46-SVG** adds view-dropout training and a reliability head, making the v45-AGF weights robust to missing views.
- **v47-temporal** post-processes the triangulated poses across time, exploiting temporal smoothness especially at sparse views.
- **v48-domain** adds domain-conditional adaptation and 3DPW actual-mode validation on top of the v47 stack.
- **v49-integration** does not add a new neural module; it adds the orchestration layer (configs, queue, ablation matrix, and feedback loop) required to ship the v46-v48 stack as a single reproducible system.

### 2.3 Self-evolution feedback loop

The v49 integration closes the self-evolution loop that v36/v37 started:

1. **v37 self-critique** predicts per-view reliability from reprojection residuals.
2. **v45-AGF / v46-SVG** consume these reliabilities as triangulation weights.
3. The **v48 domain adapter** observes per-domain residual distributions and feeds them back to the v41 dynamic-domain-weighting loss (DDWL), up-weighting the hardest domain at each epoch.
4. The updated DDWL changes the training mix, which in turn updates the v37 reliability estimator.

This loop is explicit in v49: every full-run config must set `use_v37_self_critique_view_reliability=true` when running the v46-v48 stack, and evaluation must report reliability-vs-residual correlation.

---

## 3. Concrete Code-Level Changes

### 3.1 New files

| File | Purpose |
|------|---------|
| `configs/benchmark_v49_full_stack_smoke.yaml` | Single YAML enabling v45-AGF + v46 + v47 + v48, d=64, 500 samples. |
| `configs/benchmark_v49_ablation_no_v46.yaml` | v45-AGF + v47 + v48 (no sparse-view training). |
| `configs/benchmark_v49_ablation_no_v47.yaml` | v45-AGF + v46 + v48 (no temporal head). |
| `configs/benchmark_v49_ablation_no_v48.yaml` | v45-AGF + v46 + v47 (no domain adapter). |
| `scripts/run_v49_full_stack_smoke_local_4090.sh` | Smoke launch script for the integrated stack. |
| `scripts/run_v49_ablations_local_4090.sh` | Runs the three ablations sequentially. |

### 3.2 Modified files

| File | Change |
|------|--------|
| `scripts/launch_v33_a800_queue.py` | Add a new `v49_*` block with gating logic: run v46-v48 only after v45-AGF smoke passes and v25/v45 A800 results are known. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Ensure `use_v45_adaptive_geometry_fusion`, `use_v46_sparse_view_generalization`, `use_v47_temporal_aggregation`, and `use_v48_domain_generalization` can be toggled independently without runtime errors. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add a helper `--v49_full_stack` meta-flag that expands to the validated combination of v45-v48 flags. |
| `experiments/eval_variable_views.py` | Always report `MPJPE@k` for `k ∈ {2,3,4,full}`, per-domain `MPJPE@k`, and the reliability-vs-residual correlation when v37 is active. |

### 3.3 New / updated training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_full_stack` | bool | `False` | Meta-switch enabling v45-AGF + v46 + v47 + v48 with validated defaults. |
| `v49_gate_on_v45_smoke` | bool | `True` | If true, refuse to run v49-full if no v45 smoke result exists. |
| `v49_self_evolution_feedback_weight` | float | `1.0` | Scalar on the v37→v46 reliability feedback path. |
| `v49_ddwl_feedback` | bool | `True` | Enable DDWL to feed per-domain residuals back into the domain weight schedule. |

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|------|------------|
| v46-v48 stack is heavier than v25 and still loses to v25 | Always compare against v25/v45 baseline in the same queue; abandon stack if v25 alone remains best. |
| v47 temporal over-smoothes fast motion or 3DPW in-the-wild data | Make `v47_temporal_window` configurable per domain; default to 7 for 3DPW. |
| v48 GRL/DDWL destabilizes the v46/v47 base | Freeze v25/v45/v46/v47 for the first epoch; start `v48_dg_grl_lambda=0.01`. |
| Integration YAML becomes a flag soup | Provide `use_v49_full_stack` meta-flag and a validation pre-flight that checks required dependencies. |
| A800 queue gets blocked behind unfinished v42/v43 legacy runs | Move v49 integration to the front of the queue once v45-AGF smoke passes. |
| Self-evolution feedback loop causes runaway weights | Clamp `v49_self_evolution_feedback_weight ∈ [0,1]` and use EMA for reliability estimates. |

---

## 5. Success Metrics and Recommended Experiments

### 5.1 Smoke experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_full_stack_smoke.yaml` |
| Hardware | Local RTX 4090 |
| Duration | ~1–2 hours |
| Goal | `val_MPJPE` finite, no NaN/OOM, `MPJPE@2` within 20% of `MPJPE@full` |
| Expected | ~70–80 mm full-view; sparse-view degradation ≤ 20% |

### 5.2 Full experiment (A800-D)

| Field | Value |
|-------|-------|
| Config | Reuse `v48_domain_generalization_on_v47` queue entry with `use_v49_full_stack=true` |
| Hardware | A800-D |
| Duration | ~2–3 days per run |
| Goal | Match or beat v25/v45-AGF at full views while showing sparse-view and cross-domain gains |

### 5.3 Ablation matrix

| Run | Flags | Primary question |
|-----|-------|------------------|
| v49-full | v45 + v46 + v47 + v48 | Is the integrated stack worthwhile? |
| v49-no-v46 | v45 + v47 + v48 | Does sparse-view training help? |
| v49-no-v47 | v45 + v46 + v48 | Does temporal aggregation help? |
| v49-no-v48 | v45 + v46 + v47 | Does domain adaptation help? |
| v25-full | v25 only | Strongest known baseline |
| v45-only | v25 + v45-AGF | Is v45-AGF alone enough? |

### 5.4 Success criteria

1. Smoke test passes with finite `val_MPJPE` and no NaN/OOM.
2. v49-full is within 1 mm of v45-only at full views (no regression).
3. `MPJPE@2` and `MPJPE@3` improve by ≥10% over v45-only.
4. Cross-domain gap (max per-domain MPJPE − min per-domain MPJPE) is reduced by ≥20% relative to v45-only.
5. v37 reliability-vs-residual correlation is ≥0.5 after one epoch.
6. A800 full run completes ≥1 epoch.

---

## 6. Next Experiment Queue (Prioritized)

Run in this strict order, gating each stage on the previous:

1. **v45-AGF smoke on RTX 4090** (#154 dependency)
   - Confirm v45-AGF trains and beats v25 smoke baseline.
2. **v46-SVG smoke on RTX 4090** (#160)
   - Confirm sparse-view training works; report `MPJPE@2/3/4`.
3. **v47-temporal smoke on RTX 4090** (#162)
   - Confirm temporal head improves sparse views without over-smoothing.
4. **v48-domain smoke on RTX 4090** (#164)
   - Confirm per-domain metrics are finite; 3DPW actual-mode loader works.
5. **v49-full-stack smoke on RTX 4090** (this note)
   - Single config with all flags; ablation matrix above.
6. **v49 A800 full run** (this note)
   - Only after smoke passes and GPU memory is available.
7. **v49 streaming/lite variant** (see `docs/proposals/v49_realtime_streaming.md` and `docs/swarm_iter_next/v49_lightweight_architecture_for_4090.md`)
   - After the full stack is validated; targets real-time deployment on RTX 4090.

---

## 7. Paper Story Fit

v49 turns the v46-v48 feature list into a **reproducible, gated experiment queue**. The paper can claim that the final system is not an accidental stack of modules but a deliberately integrated pipeline, where each component is validated by an ablation and the whole loop is closed through uncertainty/reliability feedback.

---

## 8. See Also

- `docs/proposals/v46_sparse_view_generalization.md`
- `docs/proposals/v47_combined_architecture.md`
- `docs/proposals/v48_domain_generalization.md`
- `docs/proposals/v31_paper_story_multiview_video_pipeline.md`
- `docs/v45_next_iteration_plan.md`
- `docs/v44_decision_plan.md`
- `docs/swarm_iter_next/v49_lightweight_architecture_for_4090.md`
- `docs/proposals/v49_realtime_streaming.md`
