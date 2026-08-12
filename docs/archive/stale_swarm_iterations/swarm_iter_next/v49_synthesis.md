# v49 Synthesis: Multi-View Video Pose with Self-Evolution Feedback

> Synthesized from 20 sub-agent notes in `docs/swarm_iter_next/v49_*.md`

## 1. Executive Summary

v49 is the next iteration of the MotionFlow-MultiView pipeline, moving from a static
`triangulate-once` architecture to a **self-evolving multi-view video system**.  The
direction keeps the v25/v45 geometry-fusion backbone and the v46–v48 stack (sparse-view
generalization, temporal aggregation, domain generalization) as the proven foundation, and
adds a small set of lightweight, gradient-safe feedback loops that use reprojection
residuals, per-view reliability, and triangulation uncertainty to refine 2-D inputs,
camera geometry, view selection, and the final 3-D pose.  All proposed v49 modules are
optional, identity-at-init where possible, and intended to be validated first on the local
RTX 4090, then scaled to A800-D full runs, with the ultimate goal of producing a coherent
ICRA/CVPR 2027 paper story: *a self-critiquing multi-view pose estimator that learns from
its own geometric consistency to generalize across sparse views, time, and domains.*

## 2. Highest-Priority Code Modules (5–7)

| # | Module | Source Notes | Why It Matters | Hard Dependencies |
|---|--------|--------------|----------------|-------------------|
| 1 | **MPJPE@k Evaluation Protocol** | `v49_evaluation_protocol_mpjpe_at_k.md` | Every v49 claim about sparse views (`MPJPE@2/3/4`), domain gap, and 3DPW actual-mode needs a single reproducible protocol. | v46 eval path (`experiments/eval_variable_views.py`) |
| 2 | **v49-Lite 4090 Baseline** | `v49_lightweight_architecture_for_4090.md` | Gives the team a fast, reproducible local baseline and unblocks iteration when A800 is queued. | v46 + v37; optional v47/v48-lite |
| 3 | **Unified Self-Evolution Feedback Head** | `v49_online_self_evolution_loop.md`, `v49_self_evolution_uncertainty_feedback.md` | Closes the prediction↔uncertainty loop inside the training graph, replacing the broken v27 TTE with a gradient-safe refinement. | v45/v46 reliability + v37; optional v47 temporal |
| 4 | **Uncertainty-Aware Triangulation + Iterative Outlier Rejection** | `v49_uncertainty_aware_triangulation.md`, `v49_outlier_view_rejection.md` | Makes triangulation robust to noisy/sparse views at the *geometric* level rather than only via learned weights. | v25/v45; v46/v37 strongly recommended |
| 5 | **Adaptive / Hard-Negative View Dropout** | `v49_view_dropout_sparse_generalization.md` | Turns the fixed v46 dropout into a curriculum that actively mines the hardest 2–3 view subsets. | v46 view dropout + v37 reliability |
| 6 | **Geometric Calibration & Triangulation (GCT)** | `v49_geometric_calibration_and_triangulation.md` | Addresses real-world camera drift; bounded camera correction avoids the v21 neural-BA divergence. | v25; v46/v37 for feedback |
| 7 | **Per-View 2D Pose Refiner** | `v49_per_view_2d_pose_extraction.md` | Opens the first stage of the pipeline so 3-D consistency can correct 2-D detection errors. | v25/v45; v46/v48 downstream |

### What we deliberately defer

* **Self-Evolving Domain Adapter (SEDA)** (`v49_domain_generalization_beyond_v48.md`) is
  valuable but *test-time self-adaptation* is higher risk and depends on a stable v48
  baseline.  Keep as Stage 3 extension.
* **Multi-Scale Temporal Aggregation beyond v47** (`v49_temporal_aggregation_beyond_v47.md`)
  is attractive, but v47 is not yet proven; we should first measure whether the existing v47
  transformer justifies its cost.
* **Multi-View Camera Embedding** (`v49_multi_view_camera_embedding_v49.md`) and
  **Cross-View Consistency Losses** (`v49_cross_view_consistency_losses.md`) are orthogonal
  enhancements; run only after the core self-evolution loop is stable.
* **WebBridge Self-Evolving Data Pipeline** (`v49_webbridge_multiview_data_pipeline.md`) is a
  data-side loop that should be added *after* the model-side loop is measured, to avoid
  conflating sources of improvement.
* **Skeleton-Aware Physical Loss v49** (`v49_skeleton_aware_physical_loss.md`) refines v40;
  low priority until v48 baseline numbers are in.
* **Scalable A800 Knobs** (`v49_scalable_architecture_for_a800.md`) is infrastructure;
  implement when the v48 full stack is ready to scale.

## 3. Contradictions, Overlaps, and Recommended Integration

### Overlapping self-evolution loops

Several sub-notes propose reprojection-residual feedback at different pipeline locations:

| Note | Proposed loop |
|------|---------------|
| `v49_online_self_evolution_loop.md` | `OnlineSelfEvolutionV49` after v47/v48, K≤2 re-triangulation steps. |
| `v49_self_evolution_uncertainty_feedback.md` | `SelfEvolutionUncertaintyFeedbackV49` updates reliability/uncertainty from reprojection/temporal/epipolar residuals. |
| `v49_uncertainty_aware_triangulation.md` | `UncertaintyAwareTriangulationV49` predicts per-view log-variance for DLT re-weighting. |
| `v49_outlier_view_rejection.md` | `IterativeOutlierViewRejectionV49` fuses v37/v46 reliability with z-score residuals. |
| `v49_geometric_calibration_and_triangulation.md` | `GeometricCalibrationTriangulationV49` refines cameras + pose from residuals. |

**Resolution:** Merge the first two into a single **Self-Evolution Feedback Head** that
predicts updated reliability/uncertainty from residuals, and feed those into a shared
**Uncertainty-Aware Triangulation** block.  The outlier-view rejection becomes the
*decision rule* inside that block (soft weights + sparse-aware guard), and GCT becomes an
optional *camera-correction* branch of the same block.  This avoids training four separate
heads that all consume reprojection residuals.

### Redundant view-selection / dropout mechanisms

* `v49_view_dropout_sparse_generalization.md` proposes adaptive dropout and hard-negative
  mining.
* v46 already has fixed Bernoulli view dropout.
* v48 proposes per-domain dropout rates.

**Resolution:** Treat v49 adaptive dropout as a **drop-in replacement for the v46 policy**,
not an additional layer.  Keep the v48 per-domain rates as a soft prior; the adaptive
policy learns on top of them.

### Redundant temporal paths

* v47 temporal aggregation exists.
* `v49_temporal_aggregation_beyond_v47.md` proposes a richer multi-scale module.
* `v49_lightweight_architecture_for_4090.md` proposes a lightweight causal Conv1D
  alternative.

**Resolution:** Keep v47 as the default.  The lightweight Conv1D becomes the v49-lite
fallback.  The multi-scale upgrade is deferred until v47 is validated.

### Conflicting domain generalization stories

* v48 uses labeled domains + FiLM/GRL/DDWL.
* `v49_domain_generalization_beyond_v48.md` proposes label-free test-time SEDA.

**Resolution:** SEDA is an *inference-time extension* of v48, not a replacement.  Train with
v48; optionally enable SEDA at test time for unknown-domain streams.

### Recommended minimal, non-redundant architecture

```
per-view 2D keypoints
        |
        v
[ PerView2DRefinerV49 ]  (optional, upstream)
        |
        v
[ v25 Multi-View Geometry Fusion ]  +  [ v45-AGF weights ]
        |
        v
[ v49 Geometric Calibration / Triangulation block ]
        |-- UncertaintyAwareTriangulationV49
        |-- IterativeOutlierViewRejectionV49
        |-- GeometricCalibrationTriangulationV49 (optional camera branch)
        |
        v
[ v46 Sparse-View Generalization ]  with  [ v49 Adaptive View Dropout ]
        |
        v
[ v47 Temporal Aggregation ]
        |
        v
[ v48 Domain Generalization ]
        |
        v
[ v49 Self-Evolution Feedback Head ]  (reprojection / temporal / epipolar residuals)
        |
        v
3D pose + updated reliability/uncertainty
```

## 4. Staged Implementation / Experiment Plan

### Stage 1 – RTX 4090 Smoke (validate components independently)

| Order | Module / Deliverable | Smoke Goal |
|-------|---------------------|------------|
| 1.1 | `MPJPE@k` protocol | Reproduce existing `MPJPE@full` within 0.1 mm; emit canonical JSON. |
| 1.2 | v49-Lite 4090 baseline | `val_MPJPE < 80 mm`, no NaN/OOM, wall-clock latency logged. |
| 1.3 | Uncertainty-aware triangulation + outlier rejection | `val_MPJPE` finite; `MPJPE@2` improves over v46-lite; no full-view regression. |
| 1.4 | Adaptive view dropout | `MPJPE@2/3` improves over v46 fixed dropout; policy entropy stays bounded. |
| 1.5 | Self-evolution feedback head | Residual-reliability Spearman > 0.3; no NaN; `MPJPE@full` within 1 mm of base. |
| 1.6 | Per-view 2D refiner | `val_MPJPE` within 5% of v48-lite; 2-D reprojection error decreases. |

All smokes use `d=64`, `clip_len=9`, `train_samples=500`, 2–5 epochs, warm-start from the
best available v46/v47/v48-lite checkpoint.

### Stage 2 – A800-D Full Runs (stack validated components)

| Order | Run | Config |
|-------|-----|--------|
| 2.1 | v49-Lite → full-scale v49 | Promote the best smoke stack to `d=128`, `n_st_layers=3`, `clip_len=13`, full WebBridge/H36M/MPI manifest. |
| 2.2 | v49 with GCT | Add camera-correction branch on top of 2.1. |
| 2.3 | v49 with per-view 2D refiner | Add 2-D refiner to the best full stack. |
| 2.4 | Scalable-A800 knobs | Enable gradient checkpointing / memory-efficient attention if needed for 2.1–2.3. |

Success gates: full-view `val_MPJPE` within 1 mm of the v48 baseline; `MPJPE@2/3` improved
by ≥3 %; no NaN/OOM across first epoch.

### Stage 3 – Ablations + Paper Story

| Order | Activity | Deliverable |
|-------|----------|---------------|
| 3.1 | Ablation matrix (disable v46/v47/v48/feedback one at a time) | `configs/ablations/v49/`, `scripts/run_v49_ablation_study_plan.sh` |
| 3.2 | Failure-analysis-driven selection | Rank components by accuracy/latency; drop components that do not meet thresholds. |
| 3.3 | Paper story packaging | `docs/paper_story_v49.md` with four-act narrative and results table. |
| 3.4 | SEDA / streaming extensions | Optional: test-time domain adaptation and real-time causal inference. |

## 5. Proposed Flags, Experiments, and Success Criteria

| Flag / Module | What it does | Smoke Experiment | Success Criteria |
|---------------|--------------|--------------------|--------------------|
| `use_v49_lite_architecture` | Enables v49-lite stack (causal Conv1D temporal, domain-conditional BN, lightweight v46) | `configs/benchmark_v49_lite_4090_smoke.yaml` | `val_MPJPE < 80 mm`; latency < 50 ms/frame; no NaN/OOM |
| `use_v49_uncertainty_aware_triangulation` + `use_outlier_view_rejection_v49` | Per-view log-variance DLT + iterative soft outlier weights | `configs/benchmark_v49_uncertainty_aware_triangulation_smoke.yaml` | `val_MPJPE < 75 mm`; `MPJPE@2` ≥ v46 baseline; finite `reproj_nll` |
| `use_v49_adaptive_view_dropout` | Learned view-keep policy + hard-negative mining | `configs/benchmark_v49_view_dropout_sparse_generalization_smoke.yaml` | `MPJPE@2/3` ≥ 5 % over fixed dropout; `policy_entropy` bounded |
| `use_online_self_evolution_v49` / `use_v49_self_evolution_uncertainty_feedback` | Residual-driven reliability/uncertainty update | `configs/benchmark_v49_self_evolution_uncertainty_feedback_smoke.yaml` | Spearman(reliability, residual) > 0.3; `MPJPE@full` within 1 mm of base |
| `use_v49_per_view_2d_refinement` | Per-view 2-D keypoint + confidence refiner | `configs/benchmark_v49_per_view_2d_refinement_smoke.yaml` | `val_MPJPE` within 5 % of v48; 2-D reproj error decreases |
| `use_geometric_calibration_triangulation_v49` | Bounded camera correction + robust re-triangulation | `configs/benchmark_v49_gct_smoke.yaml` | `val_MPJPE < 80 mm`; `U_cam` < 0.1 for >80 % views |
| `use_v49_input_format` (optional infra) | Canonical `MultiviewVideoInputFormatV49` dataclass | `configs/benchmark_v49_input_format_smoke.yaml` | `val_MPJPE` matches v46 within 0.1 mm; loader >100 samples/sec |

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **v49 re-introduces broken v27 TTE instability** | Medium | High | Keep self-evolution inside the training graph; cap iterations at K≤2; identity-at-init; clamp gates. |
| **Flag soup / combinatorial explosion** | High | Medium | Introduce a `use_v49_lite_architecture` meta-flag; enforce mutual-exclusion checks (e.g., v33 vs v49 outlier rejection). |
| **Multiple residual-based heads duplicate work** | High | Medium | Merge online self-evolution, uncertainty feedback, and outlier rejection into one triangulation/feedback block. |
| **A800 memory with stacked v46–v49** | Medium | High | Stage with v49-lite first; use scalable-A800 knobs only when needed; profile peak memory on 4090. |
| **v47/v48 baselines still unstable** | Medium | High | Gate v49 experiments on passing v48-domain smoke (#164); do not start v49 until v47/v48 are reproducible. |
| **Self-evolution feedback collapses to identity** | Medium | Medium | Auxiliary residual-reduction loss; monitor residual-reliability correlation; freeze base weights for first epoch. |
| **Adaptive dropout over-mines hard negatives and destabilizes training** | Low | Medium | Cap hard-negative probability at 0.25; enforce `min_views=2`; freeze policy for first epoch. |

## 7. Next Single Action

**Implement the canonical `MPJPE@k` evaluation protocol** (`motionflow_mv/eval/mpjpe_at_k_protocol.py` + refactor `experiments/eval_variable_views.py`).  Every subsequent v49 module depends on a single, reproducible way to measure sparse-view, per-domain, and 3DPW actual-mode performance; without this baseline metric, smoke/full comparisons across the 20 proposed directions will be noisy and irreproducible.

---

*Referenced sub-notes: `v49_online_self_evolution_loop.md`, `v49_webbridge_multiview_data_pipeline.md`, `v49_per_view_2d_pose_extraction.md`, `v49_multiview_video_input_format.md`, `v49_view_dropout_sparse_generalization.md`, `v49_scalable_architecture_for_a800.md`, `v49_outlier_view_rejection.md`, `v49_integration_and_next_experiment_queue.md`, `v49_skeleton_aware_physical_loss.md`, `v49_related_work_and_paper_story.md`, `v49_cross_view_consistency_losses.md`, `v49_multi_view_camera_embedding.md`, `v49_self_evolution_uncertainty_feedback.md`, `v49_ablation_study_plan.md`, `v49_temporal_aggregation_beyond_v47.md`, `v49_uncertainty_aware_triangulation.md`, `v49_domain_generalization_beyond_v48.md`, `v49_geometric_calibration_and_triangulation.md`, `v49_evaluation_protocol_mpjpe_at_k.md`, `v49_lightweight_architecture_for_4090.md`.*
