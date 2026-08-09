# Swarm Iteration 25 Summary — v48 Domain Generalization and 3DPW Integration

**Status:** in-progress  
**Tracking issue:** #164  
**Base branch:** `v48-domain`  
**Goal:** Improve cross-dataset generalization by integrating real-world 3DPW multi-view data and a lightweight domain-adaptive module, building on the v46/v47 stack.

## Overview

Swarm Iteration 25 carries the v48 line of work: domain generalization plus first-class 3DPW `actual`-mode support. v48 is designed as a thin extension on top of v46 Sparse-View Generalization and v47 Temporal Aggregation. It adds a `DomainAdapterV48` module, domain-conditional training, and a cross-dataset evaluation protocol, while reusing the sparse-view and temporal machinery already in place.

The iteration is tracked in `docs/swarm_iter25_action_plan.md` and the design is described in `docs/proposals/v48_domain_generalization.md`.

## Definition of done

- [ ] WebBridge 3DPW loader supports `actual` mode (real per-frame camera poses) without breaking existing manifests.
- [ ] A lightweight `DomainAdapterV48` module is added to `OmniMultiViewFusionV5`.
- [ ] Trainer supports domain-conditional training (domain embedding + optional adversarial domain loss).
- [ ] Smoke config + script created and passing on RTX 4090.
- [ ] Cross-dataset evaluation protocol reports MPJPE per dataset.
- [ ] A800 queue entry added for v48 full run.
- [x] Docs and issue #164 updated.

## Agent task assignments

| # | Type | Owner | Task | Output file(s) | Status |
|---|------|-------|------|----------------|--------|
| 1 | ANALYZE | Agent-01 | Review 3DPW loader and `convert_3dpw_multiview.py`; identify actual-mode gaps. | `docs/swarm_iter25/reports/agent01_3dpw_loader.md` | pending |
| 2 | ANALYZE | Agent-02 | Review existing domain embedding / domain weight code (v41) and propose v48 adapter design. | `docs/swarm_iter25/reports/agent02_domain_review.md` | pending |
| 3 | DESIGN | Agent-03 | Finalize `DomainAdapterV48` API and integration notes. | `docs/swarm_iter25/reports/agent03_v48_design.md` | pending |
| 4 | IMPLEMENT | Agent-04 | Add 3DPW `actual` mode support to `webbridge_mixed_dataset.py`. | `motionflow_mv/data/webbridge_mixed_dataset.py` | pending |
| 5 | IMPLEMENT | Agent-05 | Implement `DomainAdapterV48` module (FiLM / conditional BN). | `motionflow_mv/fusion/domain_adapter_v48.py` | pending |
| 6 | IMPLEMENT | Agent-06 | Wire v48 flags into `OmniMultiViewFusionV5`. | `motionflow_mv/fusion/omniview_fusion_v5.py` | pending |
| 7 | IMPLEMENT | Agent-07 | Add CLI flags and domain-loss integration in trainer. | `experiments/train_omniview_fusion_v5_webbridge_multi.py` | pending |
| 8 | IMPLEMENT | Agent-08 | Add v48 smoke config. | `configs/benchmark_v48_domain_smoke.yaml` | committed |
| 9 | IMPLEMENT | Agent-09 | Add v48 smoke run script. | `scripts/run_v48_domain_smoke_local_4090.sh` | pending |
| 10 | IMPLEMENT | Agent-10 | Write unit/integration tests. | `tests/test_domain_adapter_v48.py` | pending |
| 11 | EVAL | Agent-11 | Add per-dataset MPJPE reporting to evaluation. | `experiments/eval_omniview_fusion_v5_webbridge_multi.py` or `eval_variable_views.py` | pending |
| 12 | QUEUE | Agent-12 | Add v48 full run to A800 queue. | `scripts/launch_v33_a800_queue.py` | pending |
| 13 | DOCS | Agent-13 | Update `docs/proposals/v48_domain_generalization.md` with user guide. | `docs/proposals/v48_domain_generalization.md` | pending |
| 14 | DOCS | Agent-14 | Update `AGENTS.md` with v48 conventions. | `AGENTS.md` | pending |
| 15 | ANALYZE | Agent-15 | Propose v49 next architecture (real-time efficiency / streaming). | `docs/proposals/v49_realtime_streaming.md` | pending |
| 16 | ANALYZE | Agent-16 | Read A800 read-only results and update baseline table for v48. | `docs/swarm_iter25/reports/agent16_a800_baseline.md` | pending |
| 17 | ANALYZE | Agent-17 | Map Qwen3.8 self-evolution to v48 domain curriculum. | `docs/swarm_iter25/reports/agent17_qwen_domain.md` | pending |
| 18 | ANALYZE | Agent-18 | Review in-the-wild datasets beyond 3DPW (AIST++, 3DPW, EHF) for v49. | `docs/swarm_iter25/reports/agent18_datasets.md` | pending |
| 19 | DOCS | Agent-19 | Write `docs/swarm_iter25_summary.md` after reports land. | `docs/swarm_iter25_summary.md` | this file |
| 20 | DOCS | Agent-20 | Update GitHub issue #164 with progress and close plan. | GitHub issue #164 | pending |

> **Note:** Detailed findings from Agent-01 through Agent-18 are expected in `docs/swarm_iter25/reports/`. This summary will be updated once those reports land.

## Architecture

v48 keeps the v46/v47 backbone frozen for the first epoch, then fine-tunes jointly:

```text
Input: (B, T, V, J, 2/3) 2D keypoints + cameras
        |
        ▼
[ v46 Sparse-View Generalization ]
        |
        ├── View-dropout augmentation (domain-aware in v48)
        ├── v25 MultiViewGeometryFusionV25
        ├── v45 AdaptiveGeometryFusionV45 reliability weights
        └── Sparse-view triangulated pose P_t  (B, T, J, 3)
                |
                ▼
        [ v47 Temporal Aggregation Module ]
                |
                ├── Temporal attention over (time, joint) tokens
                ├── Domain-conditional FiLM offsets (new in v48)
                ├── View-count positional bias
                └── Residual refinement ΔP_t
                        |
                        ▼
        [ v48 Domain-Invariant Sparse-View Refinement ]
                |
                ├── Instance-normalized reliability features
                ├── Gradient-reversal domain regularization
                └── Final refined pose P'_t
```

### Key modules

- **`DomainAdapterV48`** — FiLM/conditional-BN style adapter applied after the temporal head. Learns domain-conditional offsets while keeping the rest of the network domain-invariant.
- **Domain discriminator with gradient reversal (GRL)** — encourages domain-invariant representations; target accuracy should stay near chance (0.5).
- **Dynamic Domain Weighted Loss (DDWL)** — replaces static v41 weights with learned per-domain difficulty scaling.
- **3DPW `actual`-mode loader** — loads real single-camera 3DPW sequences with per-frame intrinsics/extrinsics.

## New training flags

| Flag | Default | Purpose |
|------|---------|---------|
| `use_v48_domain_generalization` | `False` | Master toggle for v48 additions |
| `v48_dg_hidden` | `64` | Hidden dim of domain adapter |
| `v48_dg_grl_lambda` | `0.1` | Gradient-reversal layer weight |
| `v48_dg_use_domain_film` | `True` | Apply domain-conditional FiLM |
| `v48_dg_use_ddwl` | `True` | Use dynamic domain weighted loss |
| `v48_dg_ddwl_temperature` | `2.0` | DDWL temperature |
| `v48_dg_ddwl_warmup_epochs` | `1` | DDWL warmup |
| `v48_3dpw_actual_val_paths` | `None` | Paths to actual-mode val files |
| `v48_dropout_per_domain` | `{"0": 0.30, "1": 0.30, "5": 0.15}` | Per-domain view dropout |

## Evaluation protocol

Per-domain `MPJPE@k` for `k = 1, 2, 3, 4` and full views:

| Metric | Description |
|--------|-------------|
| `MPJPE@k` (per domain) | MPJPE on each domain's val split at view count k |
| `MPJPE@1` (3DPW actual) | Real monocular in-the-wild benchmark |
| `domain_gap` | Max difference in MPJPE across domains |
| `domain_discriminator_acc` | Should stay near chance (0.5) if features are domain-invariant |

Target: v48 reduces the 3DPW↔studio gap by ≥20% relative to v47 without regressing H36M/MPI/AIST full-view accuracy.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| 3DPW pseudo labels are synthetic | Use `actual` mode for real evaluation; keep pseudo as auxiliary training only. |
| GRL destabilizes training | Start with `grl_lambda=0.01`; freeze backbone for first epoch. |
| 3DPW dominates/under-represented | Combine DDWL with domain-balanced sampling. |
| Temporal head over-smoothes 3DPW motion | Use domain-conditional temporal window (shorter for 3DPW). |

## Merge plan

1. Agents commit to `v48-domain`.
2. Run smoke on RTX 4090 once v47 smoke finishes.
3. Open PR `v48-domain -> main`.
4. Review, merge, close #164.

## Next steps

- Wait for v47-temporal smoke results (#162).
- Integrate detailed findings from `docs/swarm_iter25/reports/` once they land.
- Implement 3DPW `actual`-mode loader and eval benchmark.
- Add `DomainAdapterV48` and DDWL to the v47 trainer.
- Smoke on RTX 4090 and compare per-domain `MPJPE@k` with v47.
- Queue full A800 run starting from the best v47 checkpoint.
