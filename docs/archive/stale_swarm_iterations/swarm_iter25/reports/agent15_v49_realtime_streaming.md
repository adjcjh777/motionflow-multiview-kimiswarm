# Agent-15 — v49 Real-Time / Streaming Architecture Report

**Owner:** Agent-15 (ANALYZE)  
**Branch:** `v48-domain`  
**Tracking issue:** #165 (proposed)  
**Status:** Report pushed, ready for design review  

## Summary

This report proposes the next architecture variant, **v49**, which targets **real-time efficiency and streaming inference** for the MotionFlow-MultiView stack. v49 builds on v46 Sparse-View Generalization, v47 Temporal Aggregation, and v48 Domain Generalization, adding:

1. `StreamingTemporalSmootherV49` — a causal, stateful (GRU-based) alternative to the full-clip v47 transformer, giving O(1) per-frame cost and bounded memory.
2. `DynamicViewBudgetV49` — an inference-time view-pruning policy that uses v46 reliability weights to limit the number of active cameras per frame.
3. A confidence-conditioned fast path and a concrete integration plan into `OmniMultiViewFusionV5`, the trainer, and the evaluation harness.

## Deliverable

The full proposal is located at:

```text
docs/proposals/v49_realtime_streaming.md
```

It contains the motivation, architecture diagram, module APIs, integration notes, training recipe, evaluation metrics, success criteria, and risk analysis.

## Key targets

- **30 Hz streaming:** per-frame latency ≤ 16.67 ms on RTX 4090, MPJPE within 0.5 mm of v48 batch.
- **60 Hz streaming stretch goal:** per-frame latency ≤ 8.33 ms on RTX 4090.
- **Memory:** ≤ 1 GB peak for `B=1, T=1`.
- **Accuracy guard:** identity-initialized residual gate so the module is a no-op at warm-start.

## Source files read

- `docs/swarm_iter25_action_plan.md`
- `docs/proposals/v48_domain_generalization.md`
- `AGENTS.md`
- `motionflow_mv/fusion/sparse_view_generalization_v46.py`
- `motionflow_mv/fusion/temporal_aggregation_v47.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `docs/proposals/v25_multiview_geometry_fusion.md`
- `docs/proposals/v30_adaptive_online_self_evolution.md`
- `docs/swarm_iter15/proposal_lightweight-realtime-multiview-fusion.md`
- `docs/swarm_iter_next/runtime_benchmark_report.md`
- `docs/swarm_iter18/P19_realtime_plan.md`
- `docs/swarm_iter11_real_time_efficiency_report.md`
- `docs/proposals/v19_temporal_perceiver.md`
- `docs/swarm_iter24/reports/agent03_v47_design.md`

## Tests run

No source files were modified (ANALYZE task). The only outputs are the two markdown files described above. No pytest/unit tests were required.

## Notes / questions

- A `use_gru=False` causal-conv path is included for completeness, but the GRU path is recommended for real-time because it avoids a rolling-buffer data structure.
- The proposal assumes the v46 reliability head and v47 `temporal_window` already exist, as on the current `v48-domain` branch.
