# v51 Domain-Agnostic Ensemble (DAE)

## One-line idea

Combine the existing geometry, sparse-view, temporal, domain-aware, and self-evolution branches into a single **domain-agnostic ensemble** that learns to weight per-expert predictions from geometric reliability cues alone, without requiring domain labels at inference.

## Architecture

`DomainAgnosticEnsembleV51` lives at `motionflow_mv/fusion/domain_agnostic_ensemble_v51.py` and is wired into `omniview_fusion_v5.py` when `use_v51_domain_agnostic_ensemble=True`.

It takes the following candidate 3-D pose tensors produced by upstream branches (each `(B, J, 3)`):

1. `P_geo` – v25/v45 geometry-fusion baseline.
2. `P_svg` – v46 sparse-view generalization pose.
3. `P_temp` – v47 temporal aggregation pose.
4. `P_dom` – v48 domain-conditioned pose.
5. `P_sefh` – v50 self-evolution feedback head pose.

A shared **evidence encoder** (2-layer MLP, 64-d hidden) maps a per-branch, per-joint evidence vector to logits. The evidence vector concatenates:

- Reprojection residual magnitude `(B, J)`.
- Per-view reliability entropy `(B, J)`.
- Temporal consistency score `(B, J)` (frame-to-frame joint displacement).
- Epipolar residual magnitude `(B, J)`.
- Available view count `(B, J)` broadcast.

A **gating head** outputs per-branch, per-joint weights `w_bj^k` (softmax over experts `k`) using a learned skeleton-joint embedding and the evidence vector. The final pose is the weighted sum:

```text
P_final = Σ_k w^k(P_k)
```

with weights normalized across the five experts for each joint.

To keep the module warm-startable, the gate is initialized so the v25/v45 geometry expert receives weight `≈1` and all others `≈0`. A small residual bypass adds each expert's pose scaled by `1/5` when `v51_dae_identity_bypass=True`, then gradually fades out via a learned scalar `α ∈ [0, 1]`.

## New config flags / defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_domain_agnostic_ensemble` | bool | `False` |
| `v51_dae_hidden` | int | `64` |
| `v51_dae_num_layers` | int | `2` |
| `v51_dae_dropout` | float | `0.1` |
| `v51_dae_identity_bypass` | bool | `True` |
| `v51_dae_experts` | list[str] | `["geo", "svg", "temp", "dom", "sefh"]` |
| `loss.v51_dae_loss_weight` | float | `0.005` |
| `loss.v51_dae_diversity_weight` | float | `0.001` |

## Loss term

The ensemble contributes an auxiliary loss that encourages the weighted ensemble to agree with the supervised 3-D target while keeping expert diversity positive:

```text
L_dae = λ * |P_final - P_gt|_2
        + β * (1/J) Σ_j Var_k(P_k^j)
```

where `λ = loss.v51_dae_loss_weight` and `β = loss.v51_dae_diversity_weight`. The variance term prevents all experts from collapsing to the same mean. The gate itself is trained through the weighted pose; no separate gating label is needed.

## Evaluation metric

- `val_MPJPE@full` (must remain within 1 mm of the strongest single-expert baseline, currently v46 ≈ 32.97 mm local smoke).
- `MPJPE@k` for `k = 2, 3, 4` plus full views, evaluated via `experiments/eval_variable_views.py`.
- Per-domain `MPJPE@k` on WebBridge / H36M / MPI / 3DPW actual to verify the ensemble closes the cross-domain gap without domain labels.
- **New diagnostic**: `ensemble_expert_usage` – average per-joint weight per expert on each domain, measured on validation.

## Expected MPJPE impact

- `MPJPE@2`: −3 to −5 mm versus v46 baseline.
- `MPJPE@3`: −2 to −4 mm.
- `val_MPJPE@full`: within 0.5 mm of the best single-expert baseline, since the gate is identity-at-init and the geometry expert dominates when evidence is ambiguous.
- Cross-domain 3DPW actual: −4 to −6 mm by letting the v48 domain expert and v50 self-evolution expert share weight when in-the-wild residuals are large.

## Main risk

**Risk**: The gate can overfit to the training domains and ignore the v48/v50 experts, or it can destabilize the already-good v46/v47 full-view baseline if too many experts disagree.

**Mitigation**:

1. Identity-at-init gate strongly biased toward the v25/v45 geometry expert.
2. Freeze all upstream experts for the first epoch; only the gate trains.
3. Clamp per-expert weights to `[0.05, 0.95]` to avoid hard switching.
4. Start ablations with `loss.v51_dae_loss_weight=0.001` and increase only when validation remains stable.
5. Add a smoke test that disables each expert in turn and confirms graceful degradation.

## Why this fits the v51 narrative

v50 turned the model into its own critic (SEFH). v51 takes the next step: it turns the pipeline into a **domain-agnostic committee** where geometry, sparse-view, temporal, domain-aware, and self-critique experts vote according to geometric evidence rather than domain labels. This aligns with the project's arc of making the system robust across views, time, and domains while keeping the strongest full-view baseline intact.
