# Agent-17 Report: Mapping Qwen3.8 Self-Evolution to the v48 Domain Curriculum

**Owner:** Agent-17  
**Task:** ANALYZE — Map Qwen3.8 self-evolution concepts to the v48 domain curriculum.  
**Output:** `docs/swarm_iter25/reports/agent17_qwen_domain.md`  
**Tracking issue:** #164  
**Depends on:** v47-temporal (#162), v46-SVG (#160)

---

## 1. Summary

v48 domain generalization can be understood as the next iteration of the same self-evolving loop that already drives MotionFlow-MultiView. This report maps the Qwen3.8 self-evolution principles to the **v48 domain curriculum**, with concrete per-domain dropout schedules, staged freeze/unfreeze rules, and a DDWL-driven reward signal. The core claim is:

> v48 should be trained like a self-evolving system across domains: first let the domain-invariant head and the DDWL state learn on a frozen, strong v47 backbone; then open the loop and let the whole model co-evolve under a domain-aware sparse-view curriculum.

The mapping shows that Qwen's four mechanisms — self-critique, iterative refinement, curriculum, and selection/branching — have direct counterparts in v48's domain-conditional temporal head, domain-invariant sparse-view wrapper, per-domain view-dropout curriculum, and per-domain `MPJPE@k` reward.

---

## 2. Qwen3.8 self-evolution concepts (recap)

From `docs/qwen38_selfevolution_mapping.md`, `docs/swarm_iter23/reports/agent18_qwen_selfevolution.md`, and `docs/swarm_iter24/reports/agent15_qwen_staged.md`, the four mechanisms most relevant to v48 are:

| Qwen3.8 mechanism | Meaning | Why it matters for v48 |
|---|---|---|
| **Self-critique** | The model scores its own outputs and uses the score to guide adaptation. | v48 needs to know which domains/joints are currently hardest so it can adapt loss weights and dropout schedules. |
| **Iterative refinement** | Outputs are fed back into the model for successive improvement. | v48 adds a domain-invariant refinement stage on top of v46/v47 so the model becomes less sensitive to domain-specific statistics. |
| **Curriculum / adaptive sampling** | Training difficulty is progressively increased in a controlled way. | v48 uses per-domain view dropout and temporal-window schedules to expose the model to harder domain/view combinations only after it is ready. |
| **Selection / branch decision** | The best candidate/branch is chosen by a reward/validator. | v48 is promoted to a full A800 run only if per-domain `MPJPE@k` improves over v47 and the cross-domain gap shrinks. |

---

## 3. Mapping to the v48 domain curriculum

### 3.1 Self-critique → per-domain difficulty signals in DDWL

In Qwen3.8, self-critique is a head that predicts the quality of the model's own outputs. In v48, the equivalent is the **Domain-Difficulty-Weighted Loss (DDWL)**:

- It maintains an EMA of per-domain MSE (`docs/v41_domain_loss_redesign.md`, section 2.3).
- The domain with the highest current loss is up-weighted, telling the optimizer to focus on the hardest domain right now.
- It is a *learned self-critique* of training dynamics, not just a static scalar.

```text
Input: per-domain loss history
    |
    ▼
DDWL EMA  ──>  difficulty score per domain  ──>  adaptive loss weight w_d
    │
    └── Domain with highest MSE gets the largest weight (clamped to [0.5, 2.0])
```

**Mapping to v48:** `v48_dg_use_ddwl` (proposal section *Training recipe*) reuses the v41 DDWL design and applies it to all six domains (h36m, mpi, aist, shelf, campus, 3dpw).

### 3.2 Iterative refinement → domain-invariant sparse-view wrapper

Qwen3.8 refines outputs by feeding them back through the model. In v48, the refinement loop is:

```text
P_t^(0) = v47(P_t | domain-agnostic v46/v25)
P_t^(1) = P_t^(0) + DomainInvariantSparseViewV48(P_t^(0), dataset_id, view_mask)
```

- `DomainInvariantSparseViewV48` (`docs/proposals/v48_domain_generalization.md`, section *Module API*) applies instance normalization + gradient-reversal so the reliability head depends on geometry, not domain statistics.
- The domain-conditional FiLM offsets on the v47 temporal tokens act as a second, domain-specific refinement loop inside the temporal head.

**Mapping to v48:** This is the v48 counterpart of Qwen's *action → feedback → retry* cycle: the model refines its pose estimate once more, but this time conditioned on the domain and forced to be domain-invariant.

### 3.3 Curriculum → per-domain view-dropout and temporal-window schedule

Qwen-style curriculum learning maps directly to two axes in v48:

| Axis | Studio domains (H36M/MPI/AIST) | 3DPW pseudo | 3DPW actual (eval only) |
|---|---|---|---|
| **View-dropout probability** | `p = 0.30` | `p = 0.15` | N/A (used for `MPJPE@1`) |
| **min_views** | 2 | 2 (but rarely sees <2 because V is small) | 1 |
| **Temporal window** | 7 → full clip | 7 (shorter, faster motion) | 7 or full clip (benchmark) |
| **DDWL weight** | Rises when domain is hardest | Rises when 3DPW is hardest | N/A |

This is the v48-specific instantiation of Qwen's *progressive difficulty increase*: studio domains are exposed to heavy dropout early because they have many real views to spare, while 3DPW is treated more gently because it is already view-limited and noisier.

### 3.4 Selection / reward → per-domain `MPJPE@k` as the universal reward

The reward that decides whether v48 is promoted is the sparse-view, per-domain metric:

```text
if MPJPE@1/2/3/4(v48) improves over v47 on all domains
   and domain_gap(v48) < 0.8 * domain_gap(v47)
   and domain_discriminator_acc ∈ [0.45, 0.55]:
       promote v48 to full A800 run
else:
       revisit dropout schedule, DDWL temperature, or GRL lambda
```

This closes the self-evolution loop: **design (v48 modules) → train (DDWL + curriculum) → evaluate (per-domain MPJPE@k) → select (promote or redesign)**.

---

## 4. Concrete v48 staged training recipe

The following recipe is designed to be consistent with the existing trainer (`experiments/train_omniview_fusion_v5_webbridge_multi.py`), which already supports `warm_start_freeze_epochs`, v46 curriculum dropout, and v47 temporal head freeze logic.

### Stage 0 — Warm-start from a v47 checkpoint

- Load a trained v47-SVG checkpoint.
- Freeze all v25/v45/v46/v47 parameters by default (reuse `freeze_old_params`, extended to recognize v47 prefixes).
- Keep `DomainInvariantSparseViewV48` and the DDWL EMA state trainable.

### Stage 1 — Domain-invariant head + DDWL warm-up (1 epoch)

- Keep v25/v45/v46/v47 **frozen**.
- Set `v48_dropout_per_domain` to a uniform, low value (e.g., `p=0.0` for all domains).
- Use a **local 7-frame temporal window** for all domains (`v47_temporal_window = 7`).
- Let the DDWL EMA burn in with uniform weights; do not apply adaptive weights yet.
- Goal: the domain-invariant wrapper learns to produce domain-discriminator-confused features without destabilizing the strong v47 per-frame estimates.

### Stage 2 — End-to-end fine-tuning (remaining epochs)

- **Unfreeze all** parameters (`unfreeze_all`).
- Ramp `v48_dropout_per_domain` to its domain-specific targets:
  - H36M/MPI/AIST: `p = 0.30` over the first half of training.
  - 3DPW pseudo: `p = 0.15` over the first half of training.
- Switch 3DPW to a shorter temporal window (or keep at 7) while studio domains may expand to full clip.
- Enable DDWL adaptive weights after the 1-epoch burn-in.
- Goal: the whole model adapts to domain-specific noise, view counts, and motion dynamics jointly while remaining domain-invariant at the feature level.

### 4.1 Pseudocode for the domain-curriculum training loop

```python
# Reuses existing trainer infrastructure:
#   - args.warm_start_freeze_epochs
#   - freeze_old_params / unfreeze_all
#   - v46 view-dropout curriculum (now per-domain)
#   - v47 head freeze epochs
#   - v41 DDWL state

if args.use_v48_domain_generalization:
    # Stage 1: head + DDWL warm-up
    if epoch < args.v48_head_freeze_epochs:
        freeze_v25_v45_v46_v47(model)
        effective_dropout = {d: 0.0 for d in domains}
        temporal_window = 7
        use_ddwl_weights = False
    else:
        unfreeze_all(model)
        effective_dropout = ramp_v48_dropout_per_domain(epoch, total_epochs)
        temporal_window = args.v47_temporal_window  # or domain-specific schedule
        use_ddwl_weights = True

    # Forward pass includes v48 domain-conditional wrapper
    output = model(..., dataset_id=dataset_id, view_mask=view_mask)

    # Loss includes adaptive DDWL scaling
    if use_ddwl_weights and args.v48_dg_use_ddwl:
        loss = (mse_per_sample * ddwl_weights[dataset_id]).mean()
    else:
        loss = mse_per_sample.mean()
```

### 4.2 New flags to add to the trainer

| Flag | Type | Default | Role |
|---|---|---|---|
| `v48_head_freeze_epochs` | int | 1 | Epochs to freeze v25/v45/v46/v47 and train only the v48 head/DDWL. |
| `v48_dropout_per_domain` | dict | `{"0": 0.30, "1": 0.30, "5": 0.15}` | Per-domain view-dropout probabilities. |
| `v48_ddwl_warmup_epochs` | int | 1 | Epochs of uniform DDWL weights before adaptive weighting. |
| `v48_ddwl_temperature` | float | 2.0 | DDWL temperature; higher = more uniform weights. |
| `v48_temporal_window_per_domain` | dict | `{"5": 7}` | Optional per-domain temporal window (e.g., shorter for 3DPW). |
| `v48_grl_lambda` | float | 0.01 | Gradient-reversal scale for the domain discriminator. |

---

## 5. Mapping Qwen principles to v48 risks and mitigations

| Qwen principle | v48 risk | Mitigation in domain curriculum |
|---|---|---|
| **Stable reward** | DDWL weights oscillate early and destabilize training. | 1-epoch uniform burn-in; clamp weights to `[0.5, 2.0]`; use `T >= 2.0`. |
| **Stable distribution** | 3DPW is under-represented or over-powered in mixed batches. | Domain-balanced sampling + DDWL + gentler dropout on 3DPW. |
| **Closed-loop feedback** | Domain discriminator ignores geometry and overfits to batch statistics. | Freeze GRL for the first epoch; keep `lambda` small (`0.01`); pool features only after geometry fusion. |
| **Selection/branching** | v48 regresses on studio domains while improving 3DPW. | Gate promotion on *no regression* on H36M/MPI/AIST and a reduced domain gap. |

---

## 6. Evaluation as the self-evolution reward signal

The Qwen *universal reward* has a direct equivalent in v48:

| Metric | Role as reward signal |
|---|---|
| `MPJPE@k` per domain | Primary reward; should improve or stay flat for all domains. |
| `MPJPE@1` on 3DPW actual | New in-the-wild stress-test reward; the target is a finite, improving number. |
| `domain_gap` | Secondary reward; should shrink relative to v47. |
| `domain_discriminator_acc` | Self-critique of feature invariance; should stay near 0.5. |
| `DDWL weights` | Internal self-critique of per-domain difficulty; should rise for the hardest domain. |

This reward structure is what makes the v48 loop self-evolving: the metrics from one training run directly determine whether the architecture/curriculum advances to the next stage.

---

## 7. Summary

Qwen3.8 self-evolution maps cleanly onto the v48 domain curriculum:

- **Self-critique** → DDWL EMA of per-domain loss; domain discriminator accuracy.
- **Iterative refinement** → `DomainInvariantSparseViewV48` + domain-conditional v47 temporal offsets.
- **Curriculum** → per-domain view-dropout schedule and per-domain temporal-window schedule.
- **Selection** → per-domain `MPJPE@k` and domain-gap reduction as the promotion decision.

The recommended v48 training recipe is therefore:

1. **Warm-start from v47.**
2. **Freeze base, train v48 head + DDWL** (1 epoch, full views, short temporal window, uniform DDWL).
3. **Unfreeze and co-train** under a domain-aware sparse-view curriculum (studio `p=0.30`, 3DPW `p=0.15`) with adaptive DDWL weights and domain-conditional temporal offsets.
4. **Promote** only if all-domain `MPJPE@k` is non-regressive and the 3DPW↔studio gap shrinks.

This keeps v48 minimal, stable, and aligned with the self-evolving design philosophy already established in v36–v47.

---

## References

- `docs/qwen38_selfevolution_mapping.md`
- `docs/swarm_iter23/reports/agent18_qwen_selfevolution.md`
- `docs/swarm_iter24/reports/agent15_qwen_staged.md`
- `docs/swarm_iter24/reports/agent14_v48_domain_generalization.md`
- `docs/proposals/v48_domain_generalization.md`
- `docs/v41_domain_loss_redesign.md`
- `motionflow_mv/data/view_dropout_augmentation_v46.py`
- `motionflow_mv/fusion/temporal_aggregation_v47.py`
- `motionflow_mv/fusion/sparse_view_generalization_v46.py`
- `motionflow_mv/models/domain_adaptation_wrapper.py`
