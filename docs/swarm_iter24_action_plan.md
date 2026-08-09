# Swarm Iteration 24 — v47 Temporal Aggregation on Sparse Views

**Goal:** Add a lightweight temporal aggregation head (`TemporalAggregationV47`) on top of v46 sparse-view generalization, so that sparse (2–3 view) multi-view capture approaches the accuracy of full-view capture by fusing evidence across time.

**Status:** in-progress  
**Tracking issue:** #162  
**Base branch:** `v47-temporal`  
**Depends on:** v46-SVG (#160, merged to `main`)

## Definition of done

- `motionflow_mv/fusion/temporal_aggregation_v47.py` implemented and unit-tested.
- `OmniMultiViewFusionV5` and trainer support `use_v47_temporal_aggregation`.
- Smoke config + script created and passing on RTX 4090.
- Variable-view evaluation reports `MPJPE@k` for v46 vs v47.
- A800 queue entry added for v47 full run.
- Proposal + action plan merged to `main`.

## Agent task assignments

| # | Type | Owner | Task | Output file(s) | Success check |
|---|------|-------|------|----------------|---------------|
| 1 | ANALYZE | Agent-01 | Read v45-AGF/v46 logs and predict when smoke/full results will land. | `docs/swarm_iter24/reports/agent01_status.md` | Report pushed |
| 2 | ANALYZE | Agent-02 | Review v46-SVG code and identify exact integration point for v47 head. | `docs/swarm_iter24/reports/agent02_v46_integration.md` | Report pushed |
| 3 | DESIGN | Agent-03 | Finalize `TemporalAggregationV47` API and write detailed integration notes. | `docs/swarm_iter24/reports/agent03_v47_design.md` | Report pushed |
| 4 | IMPLEMENT | Agent-04 | Implement `TemporalAggregationV47` module. | `motionflow_mv/fusion/temporal_aggregation_v47.py` | Unit tests pass |
| 5 | IMPLEMENT | Agent-05 | Wire v47 flags into `OmniMultiViewFusionV5`. | `motionflow_mv/fusion/omniview_fusion_v5.py` | Model loads with flag |
| 6 | IMPLEMENT | Agent-06 | Add CLI flags and training-loop integration in trainer. | `experiments/train_omniview_fusion_v5_webbridge_multi.py` | CLI runs |
| 7 | IMPLEMENT | Agent-07 | Add v47 smoke config. | `configs/benchmark_v47_temporal_svg_smoke.yaml` | YAML loads |
| 8 | IMPLEMENT | Agent-08 | Add v47 smoke run script. | `scripts/run_v47_temporal_svg_smoke_local_4090.sh` | Executable |
| 9 | IMPLEMENT | Agent-09 | Write unit/integration tests. | `tests/test_temporal_aggregation_v47.py` | pytest passes |
| 10 | EVAL | Agent-10 | Extend `eval_variable_views.py` to compare v46 vs v47 `MPJPE@k`. | `experiments/eval_variable_views.py` | Produces comparison CSV |
| 11 | QUEUE | Agent-11 | Add v47 full run to A800 queue. | `scripts/launch_v33_a800_queue.py` | Entry visible |
| 12 | DOCS | Agent-12 | Polish `docs/proposals/v47_combined_architecture.md` and add user guide. | `docs/proposals/v47_combined_architecture.md` | Merged |
| 13 | DOCS | Agent-13 | Update `AGENTS.md` with v47 conventions. | `AGENTS.md` | Committed |
| 14 | ANALYZE | Agent-14 | Propose v48 next architecture (domain generalization / 3DPW integration). | `docs/proposals/v48_domain_generalization.md` | Report pushed |
| 15 | ANALYZE | Agent-15 | Map Qwen3.8 self-evolution to v47 staged training (freeze/unfreeze, curriculum). | `docs/swarm_iter24/reports/agent15_qwen_staged.md` | Report pushed |
| 16 | ANALYZE | Agent-16 | Review A800 read-only historical results in `/mnt/nvme0n1p1/zhangzy/projects` and summarize baseline numbers. | `docs/swarm_iter24/reports/agent16_a800_history.md` | Report pushed |
| 17 | ANALYZE | Agent-17 | Analyze WebBridge 3DPW loader and propose v48 data expansion. | `docs/swarm_iter24/reports/agent17_3dpw_v47.md` | Report pushed |
| 18 | ANALYZE | Agent-18 | Review existing temporal modules v26/v35/v45-TGA and avoid duplication. | `docs/swarm_iter24/reports/agent18_temporal_review.md` | Report pushed |
| 19 | DOCS | Agent-19 | Write `docs/swarm_iter24_summary.md` after other reports land. | `docs/swarm_iter24_summary.md` | Committed |
| 20 | DOCS | Agent-20 | Update issue #162 with implementation progress and close plan. | GitHub issue #162 | Comment posted |

## Merge plan

1. Each agent commits to `v47-temporal` branch.
2. Run smoke test on RTX 4090 once v45-AGF medium and v46 smoke finish.
3. Open PR `v47-temporal -> main`.
4. Review, fix tests, merge.
5. Close #162 or reprioritize based on smoke results.

## Risks

- v46 smoke is pending; v47 depends on it for baseline comparison. Code can be written, but smoke results must wait.
- GPU blocked: do not run v47 smoke until v46 smoke completes.
- Avoid duplicating existing v26/v35 temporal code.
