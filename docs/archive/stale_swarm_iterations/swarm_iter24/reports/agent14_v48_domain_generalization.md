# Agent-14: v48 Domain Generalization / 3DPW Integration Proposal

**Owner:** Agent-14 (ANALYZE)  
**Tracking issue:** #162 (v47 temporal aggregation)  
**Branch:** `v47-temporal`  
**Date:** 2026-08-09

## Executive summary

v48 should be the first MotionFlow-MultiView variant explicitly optimized for **cross-domain generalization**, built on top of the v46 sparse-view and v47 temporal-aggregation stack. The central idea is to turn the model from a multi-studio pose estimator into a single model that also works on in-the-wild monocular video (3DPW), without re-training per domain. The proposal reuses existing infrastructure — the `WebBridgeMixedDataset` 3DPW pseudo/actual loaders, the `DomainAdaptationWrapper` GRL+FiLM skeleton, the v41 domain-weighted loss, and the v46 reliability head — and adds only a lightweight domain-invariant refinement stage plus a 3DPW `actual`-mode evaluation benchmark.

## What we know from v46/v47 and earlier work

| Source | Key fact | Relevance for v48 |
|--------|----------|-------------------|
| `motionflow_mv/data/webbridge_mixed_dataset.py` | `dataset_id=5` is 3DPW; skeleton already mapped to 17 joints; `return_view_mask` supported. | 3DPW pseudo can already train v46/v47 without loader changes. |
| `experiments/convert_3dpw_multiview.py` | Supports `pseudo` (4 static virtual views) and `actual` (1 moving camera, per-frame K/R/t stored). | `actual` mode is the closest in-the-wild stress test but is currently unused in training. |
| `configs/splits/webbridge_all_train_mixed.yaml` | 3DPW pseudo files are already listed. | No manifest plumbing needed for pseudo training. |
| `motionflow_mv/models/domain_adaptation_wrapper.py` | GRL domain discriminator + domain-specific FiLM adapters on the PP backbone; CPU smoke passed. | Can be reused or simplified for `OmniMultiViewFusionV5`. |
| `docs/v41_domain_loss_redesign.md` | v41 proposes adaptive per-domain, per-joint, per-term loss (DDWL). | Gives a principled way to balance 3DPW against H36M/MPI/AIST. |
| `docs/swarm_iter23/reports/agent17_3dpw_for_svg.md` | Recommends Option A (minimal) for v46 smoke, Option B (dataset-aware dropout) for full run, Option C (actual-mode eval) for in-the-wild metric. | v48 directly picks up Options B and C. |
| `docs/swarm_iter_next/v33_domain_view_curriculum.md` | Proposes `DomainAdaptiveViewCurriculum` — domain-conditional view selection. | Can be folded into v48 to handle H36M (4 views) vs MPI (14 views) vs 3DPW (1–4 views). |

## Proposed v48 scope

### 1. Goal

Train one model on the canonical H36M/MPI/AIST/3DPW mix and have it perform well on **all** of them, with special attention to the in-the-wild gap introduced by 3DPW.

### 2. Non-goals

- Replace v25/v45/v46/v47 geometry fusion or temporal aggregation.
- Introduce heavy new backbones (e.g., video transformers over raw pixels).
- Solve true zero-shot 3DPW (no labels) — v48 uses the existing 3DPW pseudo labels for training and actual labels for evaluation.

### 3. Core components

1. **Domain-conditional v47 temporal head**  
   Extend `TemporalAggregationV47` to accept `dataset_id` and learn domain-specific FiLM-like affine offsets on the temporal tokens. This lets the temporal smoother use different time constants for studio (high-fps, many views) and in-the-wild (noisier, fewer views) footage.

2. **3DPW `actual`-mode loader**  
   Add a small wrapper (`motionflow_mv/data/webbridge_3dpw_actual_loader.py`) that reads the per-frame `camera_K_frames/R_frames/t_frames` arrays produced by `convert_3dpw_multiview.py --mode actual`. It exposes single-view clips in the same `(T, 1, J, 3)` tensor layout so the v46/v47 stack can consume them unchanged.

3. **Domain-invariant sparse-view head**  
   Add `DomainInvariantSparseViewV48` in `motionflow_mv/fusion/domain_generalization_v48.py`. It wraps `SparseViewGeneralizationV46` and applies instance normalization + gradient-reversal across domains so the reliability head is encouraged to depend on geometry, not on domain-specific appearance statistics.

4. **Dataset-aware dropout curriculum (Option B from Agent-17)**  
   Extend `view_dropout_augmentation_v46.py` with per-domain `(dropout_prob, min_views)` tables. Default: 3DPW gets gentler dropout because it has fewer real views.

5. **v41-style adaptive domain loss**  
   Implement the DDWL (Domain-Difficulty-Weighted Loss) from `docs/v41_domain_loss_redesign.md` inside the v47 trainer. Maintain an EMA of per-domain MSE and up-weight 3DPW when its error is highest, preventing the model from ignoring the in-the-wild domain.

### 4. Suggested file plan

| File | Action | Purpose |
|------|--------|---------|
| `docs/proposals/v48_domain_generalization.md` | Create | Canonical proposal document. |
| `motionflow_mv/fusion/domain_generalization_v48.py` | Create | Domain-invariant sparse-view wrapper + domain-conditional FiLM. |
| `motionflow_mv/data/webbridge_3dpw_actual_loader.py` | Create | Load 3DPW `actual` files with per-frame cameras. |
| `motionflow_mv/data/view_dropout_augmentation_v46.py` | Extend | Per-dataset dropout schedule. |
| `experiments/eval_variable_views.py` | Extend | Add `--dataset_3dpw_actual` path and report `MPJPE@1` / `MPJPE@2` on real monocular sequences. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Extend | CLI flags for v48; wire DDWL; pass `dataset_id` to v47 head. |
| `motionflow_mv/fusion/temporal_aggregation_v47.py` | Extend | Optional `dataset_id` input for domain-conditional offsets. |
| `configs/benchmark_v48_domain_generalization_smoke.yaml` | Create | Smoke config for RTX 4090. |
| `scripts/run_v48_domain_generalization_smoke_local_4090.sh` | Create | Smoke launcher. |

### 5. Training recipe

1. Warm-start from the best v47 checkpoint.
2. Freeze v25/v45/v46/v47 for 1 epoch; train only the v48 domain-invariant wrapper and DDWL EMA state.
3. Unfreeze all layers and train with the mixed manifest including 3DPW pseudo.
4. Apply dataset-aware dropout: H36M/MPI/AIST `p=0.30`, 3DPW `p=0.15`.
5. Use DDWL with `T=2.0`, 1-epoch uniform warmup.
6. Validate on: (a) H36M/MPI/AIST held-out clips, (b) 3DPW pseudo val, (c) 3DPW actual single-view val.

### 6. Evaluation plan

| Metric | How computed | Target |
|--------|--------------|--------|
| `val_MPJPE` per domain | Standard eval on each domain's val split | 3DPW pseudo ≤ 1.5× MPI/H36M error; no regression on others. |
| `MPJPE@k` | `eval_variable_views.py` on each domain | v48 ≥ v47 at all k; ≥5% gain at k≤3 on H36M/MPI. |
| `MPJPE@1` on 3DPW actual | New actual-mode loader | First in-the-wild monocular benchmark. |
| Domain discriminator accuracy | Logged during training | Stays near 0.5 (domain-invariant features). |
| Domain gap | `|MPJPE_3dpw - MPJPE_mpi|` | Reduced vs. v47 baseline. |

### 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| 3DPW pseudo labels are synthetic (virtual rig) | Use `actual` mode for real monocular evaluation; consider pseudo labels as auxiliary training only. |
| Domain discriminator destabilizes pose training | Start with `lambda_domain=0.01`; freeze GRL for first epoch. |
| 3DPW dominates/under-represented in mixed batches | Use DDWL + domain-balanced sampling. |
| v47 temporal head over-smoothes 3DPW fast motion | Domain-conditional gate: smaller temporal window for 3DPW. |

### 8. Recommendation

Proceed with v48 **only after** the v47 smoke on RTX 4090 lands and shows no regression over v46. The implementation order should be:

1. Add 3DPW `actual`-mode loader and eval benchmark (low risk, high insight).
2. Implement DDWL in the v47 trainer (reuses v41 design).
3. Add domain-invariant wrapper and domain-conditional temporal offsets.
4. Smoke on RTX 4090, then queue A800 full run.

## Open questions

1. Do we have 3DPW `actual` `.npz` files already converted, or do we need to run `convert_3dpw_multiview.py --mode actual` for val/test?
2. Should v48 reuse the full `DomainAdaptationWrapper` (GRL+FiLM) or only a minimal domain-conditional FiLM on top of v47?
3. Is the v41 DDWL code already partially implemented in the trainer, or does it need to be written from scratch?
