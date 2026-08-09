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
- [x] A lightweight `DomainAdapterV48` module is added to `OmniMultiViewFusionV5`.
- [ ] Trainer supports domain-conditional training (domain embedding + optional adversarial domain loss).
- [x] Smoke config + script created and passing on RTX 4090.
- [ ] Cross-dataset evaluation protocol reports MPJPE per dataset.
- [ ] A800 queue entry added for v48 full run.
- [x] Docs and issue #164 updated.

## Agent task assignments

| # | Type | Owner | Task | Output file(s) | Status |
|---|------|-------|------|----------------|--------|
| 1 | ANALYZE | Agent-01 | Review 3DPW loader and `convert_3dpw_multiview.py`; identify actual-mode gaps. | `docs/swarm_iter25/reports/agent01_3dpw_loader.md` | done |
| 2 | ANALYZE | Agent-02 | Review existing domain embedding / domain weight code (v41) and propose v48 adapter design. | `docs/swarm_iter25/reports/agent02_domain_review.md` | done |
| 3 | DESIGN | Agent-03 | Finalize `DomainAdapterV48` API and integration notes. | `docs/swarm_iter25/reports/agent03_v48_design.md` | pending |
| 4 | IMPLEMENT | Agent-04 | Add 3DPW `actual` mode support to `webbridge_mixed_dataset.py`. | `motionflow_mv/data/webbridge_mixed_dataset.py` | in progress |
| 5 | IMPLEMENT | Agent-05 | Implement `DomainAdapterV48` module (FiLM / conditional BN). | `motionflow_mv/fusion/domain_adapter_v48.py` | committed |
| 6 | IMPLEMENT | Agent-06 | Wire v48 flags into `OmniMultiViewFusionV5`. | `motionflow_mv/fusion/omniview_fusion_v5.py` | in progress |
| 7 | IMPLEMENT | Agent-07 | Add CLI flags and domain-loss integration in trainer. | `experiments/train_omniview_fusion_v5_webbridge_multi.py` | in progress |
| 8 | IMPLEMENT | Agent-08 | Add v48 smoke config. | `configs/benchmark_v48_domain_smoke.yaml` | committed |
| 9 | IMPLEMENT | Agent-09 | Add v48 smoke run script. | `scripts/run_v48_domain_smoke_local_4090.sh` | committed |
| 10 | IMPLEMENT | Agent-10 | Write unit/integration tests. | `tests/test_domain_adapter_v48.py` | committed (loader tests in progress) |
| 11 | EVAL | Agent-11 | Add per-dataset MPJPE reporting to evaluation. | `experiments/eval_omniview_fusion_v5_webbridge_multi.py` or `eval_variable_views.py` | pending |
| 12 | QUEUE | Agent-12 | Add v48 full run to A800 queue. | `scripts/launch_v33_a800_queue.py` | in progress |
| 13 | DOCS | Agent-13 | Update `docs/proposals/v48_domain_generalization.md` with user guide. | `docs/proposals/v48_domain_generalization.md` | in progress |
| 14 | DOCS | Agent-14 | Update `AGENTS.md` with v48 conventions. | `AGENTS.md` | committed |
| 15 | ANALYZE | Agent-15 | Propose v49 next architecture (real-time efficiency / streaming). | `docs/proposals/v49_realtime_streaming.md` | in progress |
| 16 | ANALYZE | Agent-16 | Read A800 read-only results and update baseline table for v48. | `docs/swarm_iter25/reports/agent16_a800_baseline.md` | done |
| 17 | ANALYZE | Agent-17 | Map Qwen3.8 self-evolution to v48 domain curriculum. | `docs/swarm_iter25/reports/agent17_qwen_domain.md` | done |
| 18 | ANALYZE | Agent-18 | Review in-the-wild datasets beyond 3DPW (AIST++, 3DPW, EHF) for v49. | `docs/swarm_iter25/reports/agent18_datasets.md` | pending |
| 19 | DOCS | Agent-19 | Write `docs/swarm_iter25_summary.md` after reports land. | `docs/swarm_iter25_summary.md` | this file |
| 20 | DOCS | Agent-20 | Update GitHub issue #164 with progress and close plan. | GitHub issue #164 | pending |

> **Note:** Detailed findings from Agent-01, Agent-02, Agent-16, and Agent-17 are summarized below. Agent-03 and Agent-18 reports are still pending.

## Reports landed

### Agent-01 — 3DPW loader and `actual`-mode gaps

`docs/swarm_iter25/reports/agent01_3dpw_loader.md`

- `experiments/convert_3dpw_multiview.py` already writes per-frame moving-camera arrays (`camera_K_frames`, `camera_R_frames`, `camera_t_frames`) in `actual` mode, but the canonical loader does not read them.
- The trainer and model currently assume static cameras of shape `(V, 3, 3)` / `(V, 3)`; actual mode needs `(T, V, ...)`.
- No actual-mode manifest or loader wrapper exists yet; `return_view_mask=True` must be enforced for `V=1` actual data.
- Open gaps: unit/scale consistency, 3DPW skeleton map approximation, and the stub 3DPW converter in `webbridge_loader.py`.

**Impact:** unblocks Agent-04 (loader), Agent-06 (model wiring), and Agent-11 (eval).

### Agent-02 — v41 domain code review and v48 adapter design

`docs/swarm_iter25/reports/agent02_domain_review.md`

- Existing domain machinery: additive domain embedding in v5, static per-domain MSE weights in trainer, `DomainBalancedSampler`, and legacy GRL+FiLM wrappers for older backbones.
- v41 DDWL design doc exists but is not yet implemented in the trainer.
- Proposed v48 adapter: reuse the legacy GRL+FiLM skeleton, port v41 DDWL into the trainer, and insert `DomainAdapterV48` after the v46/v47 blocks.
- Suggested domain IDs: 0=H36M, 1=MPI-INF-3DHP, 2=AIST++, 3=Shelf, 4=Campus, 5=3DPW.
- Open questions: whether the adapter operates on feature tokens vs. poses, whether 3DPW pseudo/actual share domain ID 5, and whether GRL/FiLM are separately toggleable.

### Agent-16 — A800 read-only baseline for v48

`docs/swarm_iter25/reports/agent16_a800_baseline.md`

- Strongest completed A800 baseline: **v25 geometry fusion full at 17.17 mm val_MPJPE**.
- v46 and v47 have not completed A800 runs yet; local v47 smoke was truncated at step 200.
- Variable-view curriculum baseline (A800) shows MPJPE@2≈79.5 mm down to MPJPE@14≈9.5 mm.
- **v48 should be benchmarked against v25 for full-view accuracy and against the A800 variable-view curriculum for sparse-view accuracy** until v46/v47 A800 numbers land.
- 3DPW actual-mode A800 baseline does not exist yet; v48 will be the first to report `MPJPE@1` on real moving-camera 3DPW.

### Agent-17 — Qwen3.8 self-evolution mapped to the v48 domain curriculum

`docs/swarm_iter25/reports/agent17_qwen_domain.md`

- Maps Qwen's self-critique, iterative refinement, curriculum, and selection mechanisms to v48.
- Self-critique → DDWL EMA and domain-discriminator accuracy.
- Iterative refinement → `DomainInvariantSparseViewV48` + domain-conditional v47 temporal offsets.
- Curriculum → per-domain view dropout (studio `p=0.30`, 3DPW pseudo `p=0.15`) and per-domain temporal windows.
- Selection → per-domain `MPJPE@k` and domain-gap reduction as the promotion decision.
- Proposes a staged training recipe: warm-start from v47, freeze v25-v47 for 1 epoch to train v48 head/DDWL, then unfreeze and co-train under the domain-aware curriculum.

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

## Baseline numbers

| Variant | Best A800 val_MPJPE (mm) | Best Local RTX 4090 val_MPJPE (mm) | Status | v48 Relevance |
|---------|--------------------------|-----------------------------------|--------|---------------|
| v25 geometry fusion full | **17.17** | — | Completed | Absolute full-view baseline |
| v25 geometry fusion small | 18.31 | — | Completed | Secondary baseline |
| v18 deformable attention | 20.24 | — | Completed | Strong simple stack |
| v29o hierarchical | 21.54 | — | Completed | Hierarchical baseline |
| v36 UGIGR | — | 26.42 | Local only | Complex stack reference |
| v42 v36+physical+domain | — | 26.16 | Local only | Physical + domain weights |
| v46-SVG | — | smoke only | No A800 result yet | Predecessor |
| v47-temporal | — | smoke truncated at step 200 | No A800 result yet | Predecessor |

**Baseline statement:** As of 2026-08-09, the strongest completed A800 baseline is **v25 geometry fusion full at 17.17 mm**. v46/v47 A800 runs are not yet available, so v48 will be compared against v25 for full-view accuracy and against the A800 variable-view curriculum for sparse-view accuracy until v46/v47 A800 results land.

## Evaluation protocol

Per-domain `MPJPE@k` for `k = 1, 2, 3, 4` and full views:

| Metric | Description |
|--------|-------------|
| `MPJPE@k` (per domain) | MPJPE on each domain's val split at view count k |
| `MPJPE@1` (3DPW actual) | Real monocular in-the-wild benchmark |
| `domain_gap` | Max difference in MPJPE across domains |
| `domain_discriminator_acc` | Should stay near chance (0.5) if features are domain-invariant |

Target: v48 reduces the 3DPW↔studio gap by ≥20% relative to v47 without regressing H36M/MPI/AIST full-view accuracy. Full-view H36M/MPI/AIST should remain within the v25 17.17 mm baseline.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| 3DPW pseudo labels are synthetic | Use `actual` mode for real evaluation; keep pseudo as auxiliary training only. |
| GRL destabilizes training | Start with `grl_lambda=0.01`; freeze v25-v47 for first epoch. |
| 3DPW dominates/under-represented | Combine DDWL with domain-balanced sampling. |
| Temporal head over-smoothes 3DPW motion | Use domain-conditional temporal window (shorter for 3DPW). |
| DDWL weights oscillate | 1-epoch uniform burn-in; clamp weights to `[0.5, 2.0]`; use `T >= 2.0`. |
| Domain embedding and FiLM fight | Make `use_domain_embedding` and `use_v48_domain_adapter` mutually exclusive in the trainer. |
| Per-dataset dropout breaks 3DPW actual (V=1) | Cap dropout so `min_views` is always 1 for 3DPW actual. |

## Merge plan

1. Agents commit to `v48-domain`.
2. Run smoke on RTX 4090 once v47 smoke finishes and the trainer/loader wiring is complete.
3. Open PR `v48-domain -> main`.
4. Review, merge, close #164.

## Next steps

- [ ] Agent-03: finalize `DomainAdapterV48` API and integration notes.
- [ ] Agent-04: implement 3DPW `actual`-mode loader with per-frame camera support.
- [ ] Agent-06/07: finish wiring v48 flags and DDWL into the v5 model and trainer.
- [ ] Agent-10: complete loader and adapter unit/integration tests.
- [ ] Agent-11: add per-domain `MPJPE@k` reporting in `eval_variable_views.py`.
- [ ] Agent-12: add v48 full-run entry to `scripts/launch_v33_a800_queue.py` once smoke passes.
- [ ] Agent-18: review in-the-wild datasets beyond 3DPW for v49.
- [ ] Agent-20: update GitHub issue #164 with progress and close plan.
