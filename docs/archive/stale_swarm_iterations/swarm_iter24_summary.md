# Swarm Iteration 24 Summary — v47 Temporal Aggregation on Sparse Views

**Tracking issue:** #162  
**Branch:** `v47-temporal`  
**Status:** In-progress — design reports landed; implementation in flight  
**Last updated:** 2026-08-09

## Goal

Add a lightweight temporal aggregation head (`TemporalAggregationV47`) on top of the v46 sparse-view generalization (SVG) pipeline. The objective is to let 2–3 view capture configurations approach the accuracy of full-view capture by fusing evidence across time, while keeping the per-frame v25/v45/v46 foundation unchanged.

## Background

- **v45-AGF** introduced adaptive per-view/per-joint triangulation weights.
- **v46-SVG** added view-dropout augmentation and a reliability head so the model generalizes to missing views at inference.
- **v47** closes the remaining gap: with fewer simultaneous views, per-frame estimates are noisier. A shallow temporal smoother refines the triangulated 3D pose trajectory.

## Proposed architecture

```text
Input: (B, T, V, J, 2) 2D keypoints + cameras
        |
        ▼
[ v46 Sparse-View Generalization ]
        |
        ├── View-dropout augmentation (training only)
        ├── v25 MultiViewGeometryFusionV25 (per-frame)
        ├── v45 AdaptiveGeometryFusionV45 reliability weights
        └── Sparse-view triangulated pose P_t  (B, T, J, 3)
                |
                ▼
    [ v47 Temporal Aggregation Module ]
                |
                ├── Temporal attention over (time, joint) tokens
                ├── Sparse-view aware positional bias / view-count conditioning
                └── Residual refinement ΔP_t  (B, T, J, 3)
                        |
                        ▼
            Final refined pose  P_t + sigmoid(g) · ΔP_t
```

The temporal head is deliberately small:

- `d_model = 64`
- `n_heads = 4`
- `num_layers = 2`
- Optional local window (default full-clip, can set to 7 frames)
- Learnable residual gate initialized to `0.0` so the path starts as identity
- View-count conditioning via `log(n_views_t)` per frame

## Agent task roll-up

| # | Agent | Type | Task | Output |
|---|-------|------|------|--------|
| 1 | Agent-01 | ANALYZE | Predict v45/v46 smoke/full result timing | `docs/swarm_iter24/reports/agent01_status.md` |
| 2 | Agent-02 | ANALYZE | Identify v46 integration point for v47 | `docs/swarm_iter24/reports/agent02_v46_integration.md` |
| 3 | Agent-03 | DESIGN | Finalize `TemporalAggregationV47` API | `docs/swarm_iter24/reports/agent03_v47_design.md` |
| 4 | Agent-04 | IMPLEMENT | Implement `TemporalAggregationV47` | `motionflow_mv/fusion/temporal_aggregation_v47.py` |
| 5 | Agent-05 | IMPLEMENT | Wire v47 flag into `OmniMultiViewFusionV5` | `motionflow_mv/fusion/omniview_fusion_v5.py` |
| 6 | Agent-06 | IMPLEMENT | Add CLI flags / training-loop integration | `experiments/train_omniview_fusion_v5_webbridge_multi.py` |
| 7 | Agent-07 | IMPLEMENT | v47 smoke config | `configs/benchmark_v47_temporal_svg_smoke.yaml` |
| 8 | Agent-08 | IMPLEMENT | v47 smoke run script | `scripts/run_v47_temporal_svg_smoke_local_4090.sh` |
| 9 | Agent-09 | IMPLEMENT | Unit / integration tests | `tests/test_temporal_aggregation_v47.py` |
| 10 | Agent-10 | EVAL | Extend `eval_variable_views.py` for v46 vs v47 | `experiments/eval_variable_views.py` |
| 11 | Agent-11 | QUEUE | Add v47 full run to A800 queue | `scripts/launch_v33_a800_queue.py` |
| 12 | Agent-12 | DOCS | Polish proposal and user guide | `docs/proposals/v47_combined_architecture.md` |
| 13 | Agent-13 | DOCS | Update `AGENTS.md` with v47 conventions | `AGENTS.md` |
| 14 | Agent-14 | ANALYZE | Propose v48 next architecture | `docs/swarm_iter24/reports/agent14_v48_domain_generalization.md` |
| 15 | Agent-15 | ANALYZE | Map staged training ideas to v47 | `docs/swarm_iter24/reports/agent15_qwen_staged.md` |
| 16 | Agent-16 | ANALYZE | Summarize A800 historical baselines | `docs/swarm_iter24/reports/agent16_a800_history.md` |
| 17 | Agent-17 | ANALYZE | Propose 3DPW data expansion for v48 | `docs/swarm_iter24/reports/agent17_3dpw_v47.md` |
| 18 | Agent-18 | ANALYZE | Review existing v26/v35/v45-TGA temporal modules | `docs/swarm_iter24/reports/agent18_temporal_review.md` |
| 19 | Agent-19 | DOCS | Write this summary | `docs/swarm_iter24_summary.md` |
| 20 | Agent-20 | DOCS | Update issue #162 with progress | GitHub issue #162 |

## Detailed design (Agent-03)

`TemporalAggregationV47` is a pose-level post-triangulation smoother. It is orthogonal to existing v26/v35 temporal modules because it operates only on the final 3D pose trajectory `(B, T, J, 3)`.

### Module API

```python
class TemporalAggregationV47(nn.Module):
    def __init__(
        self,
        n_joints: int = 17,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        temporal_window: Optional[int] = None,
        dropout: float = 0.1,
        residual_gate_init: float = 0.0,
        use_view_count_conditioning: bool = True,
    ) -> None: ...

    def forward(
        self,
        poses_3d: torch.Tensor,        # (B, T, J, 3)
        view_mask: torch.Tensor,       # (B, T, V)
        clip_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:                # (B, T, J, 3)
```

### Internal flow

1. Optionally concatenate `log(max(n_views_t, 1))` to each joint token.
2. Flatten `(T, J)` to `T*J` tokens and project to `d_model`.
3. Add learned/sinusoidal positional embeddings.
4. Run `num_layers` transformer encoder layers.
5. Project back to 3, reshape to `(B, T, J, 3)`, and add a gated residual: `poses_3d + sigmoid(g) * ΔP_t`.

### Integration points

- Inserted into `OmniMultiViewFusionV5.forward` after the v46 triangulation output.
- `view_mask` is reshaped from whatever form the v46 path produces back to `(B, T, V)`.
- `clip_mask` marks valid frames; masked tokens are excluded from attention.
- When `temporal_window` is set, a band-diagonal attention mask keeps latency bounded for streaming.

### New training flags (core)

- `use_v47_temporal_aggregation` (default `False`)
- `v47_temporal_d_model` (default `64`)
- `v47_temporal_n_heads` (default `4`)
- `v47_temporal_num_layers` (default `2`)
- `v47_temporal_window` (default `None`)
- `v47_temporal_dropout` (default `0.1`)
- `v47_temporal_loss_weight` (default `0.01`)
- `v47_use_view_count_conditioning` (default `True`)

## Staged training recipe (Agent-15)

Agent-15 maps the Qwen3.8 self-evolution principles to a concrete v47 staged training recipe:

| Stage | Frozen modules | View dropout | Temporal window | Smoothness loss |
|-------|----------------|--------------|-------------------|-----------------|
| 0 — Warm-start | Load v46-SVG checkpoint | — | — | — |
| 1 — Head warm-up | v25/v45/v46 frozen | `0.0` | `7` frames | `0.0` |
| 2 — End-to-end | All unfrozen | Ramp to target (`0.1 → 0.3`) | Expand to full clip | `0.01` |

Additional flags suggested by the staged recipe:

- `v47_head_freeze_epochs` (default `1`)
- `v47_temporal_full_clip_after_warmup` (default `False`)
- `v47_temporal_loss_weight_start` (default `0.0`)
- `v47_curriculum_window` (default `True`)

The smoothness loss is computed on the refined output:

```python
loss += v47_temporal_loss_weight * mean(|P_t - P_{t-1}|)
```

Promotion rule: v47 is promoted only if `MPJPE@2/3` improves by ≥5% over v46 **and** `MPJPE@full` does not regress.

## v48 next architecture (Agent-14)

Agent-14 proposes v48 as the first cross-domain generalization variant, built on the v46/v47 stack:

- **Goal:** One model trained on H36M/MPI/AIST/3DPW that works in-studio and in-the-wild.
- **Key additions:**
  - Domain-conditional v47 temporal head with FiLM-like offsets per domain.
  - 3DPW `actual`-mode loader for real monocular evaluation.
  - `DomainInvariantSparseViewV48` wrapper around v46 with instance normalization + gradient reversal.
  - Dataset-aware view dropout (gentler on 3DPW, which has fewer real views).
  - v41-style Domain-Difficulty-Weighted Loss (DDWL).
- **Recommendation:** Proceed only after v47 smoke lands with no regression over v46.

## Training and evaluation plan

1. Start from a v46-SVG checkpoint or train v46 from scratch.
2. Freeze v25/v45/v46 weights for the first epoch; let the temporal head warm up on stable per-frame poses.
3. Unfreeze and fine-tune end-to-end with the same view-dropout curriculum.
4. Apply the temporal smoothness loss on the refined output.

Evaluation extends `experiments/eval_variable_views.py` to report `MPJPE@k` for `k ∈ {2, 3, 4, full}` for both v46 and v47, plus relative improvement and `temporal_jerk@k`.

**Target:** v47 improves v46 by **≥5% MPJPE at k ≤ 3** with no regression at full views and lower temporal jerk.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| v46 smoke still pending | Do not run v47 smoke until v46 smoke completes and frees the RTX 4090. |
| Temporal head over-smoothes fast motion | Residual gate starts at 0; clip-masked attention; tune `v47_temporal_loss_weight`. |
| Added latency for streaming | Default to a 7-frame local window; full-clip attention is optional. |
| Duplication of v26/v35 temporal code | v47 operates on output 3D poses, not mid-level ray tokens, keeping it distinct. |
| Training instability on unfreeze | Staged unfreeze: temporal head first, then all layers. |

## Current status

- `docs/swarm_iter24_action_plan.md` and `docs/proposals/v47_combined_architecture.md` are in place.
- Design reports have landed:
  - `docs/swarm_iter24/reports/agent03_v47_design.md` — module API and integration contract
  - `docs/swarm_iter24/reports/agent15_qwen_staged.md` — staged training recipe
  - `docs/swarm_iter24/reports/agent14_v48_domain_generalization.md` — v48 proposal
- Implementation files have started appearing (untracked/staged):
  - `motionflow_mv/fusion/temporal_aggregation_v47.py`
  - `tests/test_temporal_aggregation_v47.py`
- The `docs/swarm_iter24/reports/` directory is otherwise empty for Agents 01, 02, 16–18.
- The first v47 smoke test on RTX 4090 remains blocked until v45-AGF medium and v46-SVG smoke finish.

## Next steps

1. Wait for v45-AGF medium and v46-SVG smoke results (#160).
2. Land remaining ANALYZE/DESIGN reports in `docs/swarm_iter24/reports/`.
3. Complete `TemporalAggregationV47` implementation, wiring, tests, and smoke config/script.
4. Run v47 smoke on RTX 4090 using the staged recipe (head-only epoch, then end-to-end).
5. Compare `MPJPE@k` against v46 and queue a full A800 run if the smoke target (< 75 mm) is met.
6. Merge `v47-temporal -> main` via PR and close #162 or reprioritize based on smoke results.
