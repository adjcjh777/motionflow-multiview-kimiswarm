# Swarm Iteration 23 — v46 Sparse-View Generalization (SVG)

**Goal:** Add sparse-view generalization to `OmniMultiViewFusionV5` so the model trains robustly with missing/dropped camera views and evaluates on variable-view subsets. This is the next concrete step toward the ICRA/CVPR 2027 paper story: *practical multi-view video capture without a fixed rig*.

**Status:** in-progress
**Tracking issue:** #160
**Base branch:** `v46-svg`
**Depends on:** v45-AGF (#158), v25 all-train baseline (#154)

## Definition of done

- `motionflow_mv/fusion/sparse_view_generalization_v46.py` implemented and unit-tested.
- View-dropout augmentation in the training loop.
- `OmniMultiViewFusionV5` and trainer accept new `use_v46_sparse_view_generalization` flags.
- Smoke config + script created; smoke run completes on RTX 4090.
- Variable-view evaluation extended to report MPJPE@k.
- A800 queue entry added.
- Proposal + action plan merged to `main` via PR.

## Agent task assignments

| # | Type | Owner | Task | Output file(s) | Success check |
|---|------|-------|------|----------------|---------------|
| 1 | ANALYZE | Agent-01 | Read `outputs/v45_agf_medium_local_4090.log` and summarize current epoch/loss/val trend; predict final val. | `docs/swarm_iter23/reports/agent01_v45_status.md` | Report pushed to branch |
| 2 | ANALYZE | Agent-02 | SSH read-only check of A800 v25 all-train baseline tmux/log; estimate first val time. | `docs/swarm_iter23/reports/agent02_a800_status.md` | Report pushed to branch |
| 3 | ANALYZE | Agent-03 | Review existing `VariableViewSetAggregator` and `variable_view_training` code; identify integration points for v46. | `docs/swarm_iter23/reports/agent03_variable_view_review.md` | Report pushed to branch |
| 4 | ANALYZE | Agent-04 | Review v45-AGF `AdaptiveGeometryFusionV45`; identify how reliability weights can be reused. | `docs/swarm_iter23/reports/agent04_v45_reuse.md` | Report pushed to branch |
| 5 | DESIGN | Agent-05 | Produce detailed v46-SVG module API and integration notes. | `docs/swarm_iter23/reports/agent05_v46_design.md` | Report pushed to branch |
| 6 | IMPLEMENT | Agent-06 | Implement `SparseViewGeneralizationV46` module. | `motionflow_mv/fusion/sparse_view_generalization_v46.py` | Unit test passes |
| 7 | IMPLEMENT | Agent-07 | Implement view-dropout augmentation helper. | `motionflow_mv/data/view_dropout_augmentation_v46.py` | Unit test passes |
| 8 | IMPLEMENT | Agent-08 | Wire v46 flags into `OmniMultiViewFusionV5` and call the module. | `motionflow_mv/fusion/omniview_fusion_v5.py` | Model loads with/without flag |
| 9 | IMPLEMENT | Agent-09 | Add CLI flags and training-loop dropout call in trainer. | `experiments/train_omniview_fusion_v5_webbridge_multi.py` | `--use_v46_sparse_view_generalization` runs |
| 10 | IMPLEMENT | Agent-10 | Add v46-SVG smoke config. | `configs/benchmark_v46_svg_smoke.yaml` | Config loads |
| 11 | IMPLEMENT | Agent-11 | Add v46-SVG smoke run script. | `scripts/run_v46_svg_smoke_local_4090.sh` | Script is executable |
| 12 | IMPLEMENT | Agent-12 | Write unit/integration tests. | `tests/test_sparse_view_generalization_v46.py` | `pytest` passes |
| 13 | EVAL | Agent-13 | Extend variable-view evaluation to report MPJPE@k. | `experiments/eval_variable_views.py` | Produces CSV/JSON with @k |
| 14 | QUEUE | Agent-14 | Add v46-SVG full run to A800 queue. | `scripts/launch_v33_a800_queue.py` | Entry visible in queue |
| 15 | DOCS | Agent-15 | Polish proposal and write user-facing docs. | `docs/proposals/v46_sparse_view_generalization.md` | Merged via PR |
| 16 | DOCS | Agent-16 | Update AGENTS.md with v46 conventions. | `AGENTS.md` | Committed to branch |
| 17 | ANALYZE | Agent-17 | Analyze WebBridge 3DPW loader and propose v46 data integration. | `docs/swarm_iter23/reports/agent17_3dpw_for_svg.md` | Report pushed |
| 18 | ANALYZE | Agent-18 | Map Qwen3.8 self-evolution concepts to our design-train-evaluate loop. | `docs/swarm_iter23/reports/agent18_qwen_selfevolution.md` | Report pushed |
| 19 | ANALYZE | Agent-19 | Propose v47 combined architecture (v46-SVG + temporal aggregation). | `docs/proposals/v47_combined_architecture.md` | Report pushed |
| 20 | DOCS | Agent-20 | Write `docs/swarm_iter23_summary.md` after other reports land. | `docs/swarm_iter23_summary.md` | Committed to branch |

## Merge plan

1. Each agent commits to `v46-svg` branch.
2. Run smoke test on RTX 4090 when GPU is free.
3. Open PR `v46-svg -> main`.
4. Review, fix tests, merge.
5. Close #160 as completed or reprioritize based on smoke results.

## Risks

- **GPU blocked:** 4090 is running v45-AGF medium. Do not start v46 smoke until it finishes.
- **File conflicts:** Agent-08, Agent-09, and Agent-13 may touch related files. Each edits a distinct file; conflicts will be resolved during PR review.
- **Overdesign:** v46 must be a small, isolated module. If it grows beyond the plan, flag in #160.
