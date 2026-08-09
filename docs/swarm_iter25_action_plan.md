# Swarm Iteration 25 — v48 Domain Generalization and 3DPW Integration

**Goal:** Improve cross-dataset generalization by integrating real-world 3DPW multi-view data and a lightweight domain-adaptive module, building on the v46/v47 stack.

**Status:** in-progress  
**Tracking issue:** #164  
**Base branch:** `v48-domain`  
**Depends on:** v46-SVG (#160), v47-temporal (#162)

## Definition of done

- WebBridge 3DPW loader supports `actual` mode (real per-frame camera poses) without breaking existing manifests.
- A lightweight `DomainAdapterV48` module is added to `OmniMultiViewFusionV5`.
- Trainer supports domain-conditional training (domain embedding + optional adversarial domain loss).
- Smoke config + script created and passing on RTX 4090.
- Cross-dataset evaluation protocol reports MPJPE per dataset.
- A800 queue entry added for v48 full run.
- Docs and issue #164 updated.

## Agent task assignments

| # | Type | Owner | Task | Output file(s) | Success check |
|---|------|-------|------|----------------|---------------|
| 1 | ANALYZE | Agent-01 | Review 3DPW loader and `convert_3dpw_multiview.py`; identify actual-mode gaps. | `docs/swarm_iter25/reports/agent01_3dpw_loader.md` | Report pushed |
| 2 | ANALYZE | Agent-02 | Review existing domain embedding / domain weight code (v41) and propose v48 adapter design. | `docs/swarm_iter25/reports/agent02_domain_review.md` | Report pushed |
| 3 | DESIGN | Agent-03 | Finalize `DomainAdapterV48` API and integration notes. | `docs/swarm_iter25/reports/agent03_v48_design.md` | Report pushed |
| 4 | IMPLEMENT | Agent-04 | Add 3DPW `actual` mode support to `webbridge_mixed_dataset.py`. | `motionflow_mv/data/webbridge_mixed_dataset.py` | Loader tests pass |
| 5 | IMPLEMENT | Agent-05 | Implement `DomainAdapterV48` module (FiLM / conditional BN). | `motionflow_mv/fusion/domain_adapter_v48.py` | Unit tests pass |
| 6 | IMPLEMENT | Agent-06 | Wire v48 flags into `OmniMultiViewFusionV5`. | `motionflow_mv/fusion/omniview_fusion_v5.py` | Model loads |
| 7 | IMPLEMENT | Agent-07 | Add CLI flags and domain-loss integration in trainer. | `experiments/train_omniview_fusion_v5_webbridge_multi.py` | CLI runs |
| 8 | IMPLEMENT | Agent-08 | Add v48 smoke config. | `configs/benchmark_v48_domain_smoke.yaml` | YAML loads |
| 9 | IMPLEMENT | Agent-09 | Add v48 smoke run script. | `scripts/run_v48_domain_smoke_local_4090.sh` | Executable |
| 10 | IMPLEMENT | Agent-10 | Write unit/integration tests. | `tests/test_domain_adapter_v48.py` | pytest passes |
| 11 | EVAL | Agent-11 | Add per-dataset MPJPE reporting to evaluation. | `experiments/eval_omniview_fusion_v5_webbridge_multi.py` or `eval_variable_views.py` | Produces per-dataset CSV |
| 12 | QUEUE | Agent-12 | Add v48 full run to A800 queue. | `scripts/launch_v33_a800_queue.py` | Entry visible |
| 13 | DOCS | Agent-13 | Update `docs/proposals/v48_domain_generalization.md` with user guide. | `docs/proposals/v48_domain_generalization.md` | Merged |
| 14 | DOCS | Agent-14 | Update `AGENTS.md` with v48 conventions. | `AGENTS.md` | Committed |
| 15 | ANALYZE | Agent-15 | Propose v49 next architecture (real-time efficiency / streaming). | `docs/proposals/v49_realtime_streaming.md` | Report pushed |
| 16 | ANALYZE | Agent-16 | Read A800 read-only results and update baseline table for v48. | `docs/swarm_iter25/reports/agent16_a800_baseline.md` | Report pushed |
| 17 | ANALYZE | Agent-17 | Map Qwen3.8 self-evolution to v48 domain curriculum. | `docs/swarm_iter25/reports/agent17_qwen_domain.md` | Report pushed |
| 18 | ANALYZE | Agent-18 | Review in-the-wild datasets beyond 3DPW (AIST++, 3DPW, EHF) for v49. | `docs/swarm_iter25/reports/agent18_datasets.md` | Report pushed |
| 19 | DOCS | Agent-19 | Write `docs/swarm_iter25_summary.md` after reports land. | `docs/swarm_iter25_summary.md` | Committed |
| 20 | DOCS | Agent-20 | Update GitHub issue #164 with progress and close plan. | GitHub issue #164 | Comment posted |

## Merge plan

1. Agents commit to `v48-domain`.
2. Run smoke on RTX 4090 once v47 smoke finishes.
3. Open PR `v48-domain -> main`.
4. Review, merge, close #164.

## Risks

- v48 touches data loader; must not break H36M/MPI manifests.
- 3DPW `actual` data may be large; smoke should use a small subset.
