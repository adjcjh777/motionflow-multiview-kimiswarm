# v49 Ablation Study Plan

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `ablation`, `P1-next`  
**Tracking issue:** #167 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain), #166 (v49-lite)  

---

## 1. Problem Statement

The v46–v49 stack is composed of several independently motivated modules:

- **v46 Sparse-View Generalization** (view dropout + per-view reliability head)
- **v47 Temporal Aggregation** (post-triangulation temporal transformer)
- **v48 Domain Generalization** (domain-conditional FiLM/GRL + DDWL)
- **v37-style Self-Evolution Feedback** (per-view reliability from reprojection residuals)
- **v49 Real-Time Streaming / Lightweight Architecture** (causal streaming smoother, dynamic view budget)

We currently have *strong local signals* for each component, but no staged ablation tells us which pieces are actually necessary for the final ICRA/CVPR 2027 story.  Running the full v48 model and removing one module at a time is expensive and noisy, and the v49 candidates (streaming vs. clip-based, lightweight vs. full) have not been compared under the same data and evaluation protocol.

This subtopic defines a **minimal, staged ablation plan** that:

1. Starts from a strong v48 + v37-feedback baseline.
2. Systematically removes or replaces one component at a time.
3. Compares v49 streaming/lightweight alternatives against the full v48 baseline.
4. Produces a single ranked list of components that justify their compute/memory cost.

---

## 2. Proposed Approach and Fit with v46–v48 / v49

We run a **fixed master config** on a small WebBridge mixed manifest and ablate by toggling existing flags.  All ablations share the same v25 geometry-fusion backbone, optimizer, and early-stopping rule so that differences are attributable only to the component under test.

### Ablation groups

| Group | Variants | What it tells us |
|-------|----------|------------------|
| **A. Component ablations from v48** | Full v48 → no v48 domain → no v47 temporal → no v46 dropout → no v37 feedback | Which layers are load-bearing for accuracy vs. cross-domain generalization |
| **B. v46 reliability ablations** | v46 reliability head vs. uniform triangulation vs. oracle (GT visibility) mask | Whether learned reliability is better than simple masking |
| **C. v47 temporal ablations** | v47 transformer vs. v49-lite causal Conv1D vs. no temporal smoothing | Whether the heavy transformer is needed, or the lightweight causal head suffices |
| **D. v48 domain ablations** | FiLM/GRL on/off, DDWL on/off, per-domain dropout on/off | Which domain pieces close the 3DPWstudio gap |
| **E. v49 real-time ablations** | v49 streaming GRU vs. clip v47; dynamic view budget `max_views=2,3,4` | Latency/accuracy Pareto frontier for deployment |
| **F. Self-evolution ablations** | v37 feedback loop open vs. closed; update every frame vs. every K frames | Whether the uncertainty→reliability loop stabilizes or destabilizes training |

### How it fits into the broader pipeline

```text
multi-view video
    |
    ▼
v25 Multi-View Geometry Fusion  (frozen backbone)
    |
    ▼
v46 Sparse-View Generalization  ── ablate: dropout / reliability / uniform tri
    |
    ▼
v37 Self-Critique Feedback  ────── ablate: open/closed loop
    |
    ▼
v47 Temporal Aggregation  ─────── ablate: transformer / v49-lite causal / none
    |
    ▼
v48 Domain Generalization  ────── ablate: FiLM/GRL / DDWL / per-domain dropout
    |
    ▼
v49 Streaming vs. Batch Output  ─ ablate: GRU smoother / dynamic view budget
    |
    ▼
3-D pose
```

The output of this ablation study directly informs the **final v49 architecture**: only components that survive the ablation (significant accuracy gain or acceptable latency trade-off) are kept in the production checkpoint.

---

## 3. Concrete Code-Level Changes

### New files

- `scripts/run_v49_ablation_study_plan.sh`
  - Orchestrates the full ablation matrix on RTX 4090 or A800-D.
  - Accepts a `--stage smoke|full` flag and loops over YAML snippets in `configs/ablations/v49/`.

- `configs/ablations/v49/` (new directory)
  - `baseline_v48_v37.yaml` — full v48 + v37-feedback master config.
  - `ablate_v48_domain.yaml`
  - `ablate_v47_temporal.yaml`
  - `ablate_v46_dropout.yaml`
  - `ablate_v46_reliability.yaml`
  - `ablate_v37_feedback.yaml`
  - `v49_lite_temporal.yaml`
  - `v49_streaming_gru.yaml`

- `experiments/eval_ablation_v49.py`
  - Wraps `eval_variable_views.py` for a list of checkpoints and produces a single `ablation_matrix.json`:
    - `MPJPE@k` for `k = 2, 3, 4, full`
    - `MPJPE@1` on 3DPW actual
    - `domain_gap`
    - `latency_ms_per_frame` (RTX 4090)
    - `peak_mem_MB`

### Modified files

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add a small ablation helper so flags are mutually documented:
    - `v49_ablate_disable_v46_dropout` (bool, default `False`)
    - `v49_ablate_disable_v46_reliability` (bool, default `False`)
    - `v49_ablate_disable_v47_temporal` (bool, default `False`)
    - `v49_ablate_disable_v48_domain` (bool, default `False`)
    - `v49_ablate_disable_v37_feedback` (bool, default `False`)
  - These flags override the corresponding `use_*` flags in a deterministic order inside the model constructor.

- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Expose the five `v49_ablate_disable_*` flags on the CLI.
  - Ensure the ablation flags are saved in the run metadata so results are reproducible.

- `experiments/eval_variable_views.py`
  - Add `--report_ablation_row` flag that prints a single CSV row: `variant,MPJPE@2,MPJPE@3,MPJPE@4,MPJPE@full,MPJPE@1_3dpw,latency_ms,mem_MB`.

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|------|------------|
| Removing a component uncovers hidden dependencies (e.g. v48 expects v47 output shape). | Run a quick import/forward smoke for each YAML before the full matrix. |
| Ablations are noisy on small smoke data. | Fix seed, use the same 500-sample manifest, and run each variant once on the same split. |
| v37 feedback loop causes NaN when opened/closed mid-ablation. | Initialize feedback weight to `0.0` and gate it with `v49_ablate_disable_v37_feedback`. |
| Dynamic view budget (`max_views=2`) can produce degenerate triangulation. | Enforce `min_views=2` and skip variants that fail a NaN check. |
| Full ablation matrix is too expensive for A800 priority queue. | Run the full matrix on RTX 4090 with `train_samples=4000`; A800 only for the two best variants. |

---

## 5. Success Metrics and Recommended Experiments

### Smoke experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Script | `bash scripts/run_v49_ablation_study_plan.sh --stage smoke` |
| Hardware | Local RTX 4090 (24 GiB) |
| Config | `configs/benchmark_v48_domain_smoke.yaml` as master, 500 samples, 5 epochs |
| Goal | Every variant finishes with finite `val_MPJPE`; no NaN/OOM |
| Expected | Baseline ~70–80 mm; each ablation delta is measurable (>2 mm or >10% latency) |

### Full experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Script | `bash scripts/run_v49_ablation_study_plan.sh --stage full` |
| Hardware | Local RTX 4090 |
| Config | `configs/benchmark_v48_domain_smoke.yaml` scaled to `train_samples=4000`, `clip_len=13`, 10 epochs |
| Goal | Rank components by `val_MPJPE@full`, `MPJPE@2`, `3DPW actual MPJPE@1`, and `latency_ms` |
| Expected | Full v48+v37 remains best on accuracy; v49-lite causal head is best on latency/accuracy Pareto |

### Decision thresholds

A component is kept if **any** of the following hold:

- It improves `val_MPJPE@full` by ≥2 mm vs. removing it.
- It improves `MPJPE@2` by ≥5%.
- It reduces `3DPW actual MPJPE@1` by ≥5%.
- It reduces per-frame latency by ≥20% with ≤1 mm accuracy regression.

Otherwise it is dropped from the final v49 architecture.

---

## 6. Self-Evolution Feedback Loop

The ablation plan explicitly tests the **self-evolution loop** inspired by v36/v37:

1. **Per-view uncertainty:** the v37 self-critique head predicts a reliability score from the reprojection residual of each view.
2. **Closed-loop re-weighting:** this score scales the v46 sparse-view reliability weights before triangulation.
3. **Iterative refinement:** the triangulated pose is fed into the temporal head; the refined pose’s residual is used to update the v37 reliability estimator.
4. **Ablation:** `v49_ablate_disable_v37_feedback = True` breaks the loop (open-loop), so the v37 head is trained but not allowed to modify v46 weights.

We expect the closed loop to help most at sparse views (`MPJPE@2`), where per-view quality is most ambiguous, and we expect it to be cheap enough to keep even in the v49-lite variant.

---

## 7. Next Steps

1. Wait for v48-domain smoke (#164) and v49-lite smoke (#166) to land.
2. Add the five `v49_ablate_disable_*` flags to `OmniMultiViewFusionV5` and the trainer.
3. Write the YAML ablation configs and the orchestrator script.
4. Run the smoke ablation matrix on RTX 4090.
5. Promote the two best component combinations to the full 4090 run and, if warranted, to A800-D.
