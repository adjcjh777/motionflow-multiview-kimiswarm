# P20 Synthesis — Next-iteration action plan for swarm-iter18-omniview

**Branch:** `feat/swarm-iter18-omniview`  
**Date:** 2026-08-07  
**Status:** Living action plan (updates as iter18/iter19 experiments land)  

## 1. Context: what the iter18 swarm produced

The `docs/swarm_iter18/P*.md` files read for this synthesis are:

- `P02_omniview_arch.md` — architectural design for **OmniMultiViewFusion v2**, a single model that fuses principal-point/focal correction, visibility gating, graph-joint attention, spatiotemporal (T×V×J) attention, and uncertainty-weighted triangulation.
- `P11_paper_story.md` — narrative justification for moving from the current Bayesian Tri v2 ensemble (8.35 mm MPJPE on MPI-INF-3DHP S2/Seq1) to a unified architecture, with updated paper sections and risk register.

Both converge on the same strategic direction: **stop stacking isolated modules and build one publication-ready fusion backbone**.

## 2. Strategic goal

Deliver a single model, **OmniMultiViewFusion v2**, that:

| Target | Metric |
|--------|--------|
| Clean accuracy | **≤ 8.0 mm** MPJPE on MPI-INF-3DHP S2/Seq1 (single model, not ensemble) |
| PA-MPJPE | **≤ 5.0 mm** |
| Occlusion 30% | **≤ 13 mm** (vs. current 16.99 mm) |
| View dropout 30% | **≤ 13 mm** (vs. current 18.15 mm) |
| rot_0.5° | **≤ 14 mm** (vs. current 16.89 mm) |
| Variable views k=4 | **≤ 25 mm** |
| Runtime | ≥ 100 clips/s on RTX 4090 |

These numbers close the gap between the current ensemble and a **camera-ready ICRA/CVPR 2027 method section**.

## 3. Key themes extracted from P02 and P11

1. **Unification over ensemble.** A single differentiable module is easier to explain, diagram, and ship than a collection of variants.
2. **Visibility before fusion.** Predict per-view, per-joint visibility multipliers early and use them as masks in view-attention and triangulation.
3. **Geometry in the loop.** Intrinsic self-calibration, ray-aware embedding, and uncertainty-weighted Gauss-Newton triangulation remain the core differentiators.
4. **Graph-structured joint reasoning.** Replace dense joint self-attention with anatomical skeleton-graph edges to improve occlusion robustness.
5. **Warm-start discipline.** Freeze the strong PP-corrected encoder for 5 epochs; train only the new heads first, then unfreeze.
6. **Evidence before scaling.** Every new component must pass CPU smoke and a small d=48/10-epoch GPU smoke before a full run is queued.

## 4. Next-iteration action plan

### Phase 0 — Prototype hardening (CPU, now)

| # | Action | Artifact | Validation | Gate |
|---|--------|----------|------------|------|
| 0.1 | Land `OmniMultiViewFusion` skeleton in `motionflow_mv/fusion/omniview_fusion_v2.py` (or an isolated prototype) and wire to trainer interface. | `motionflow_mv/fusion/omniview_fusion_v2.py` | `python -m py_compile` passes; forward pass with `(B=2,T=5,V=4,J=17,d=48)` returns correct shapes and finite gradients. | No NaNs; output shapes match spec. |
| 0.2 | Add CPU smoke test that exercises visibility head, graph edge rebuild, and uncertainty head. | `tests/test_omniview_fusion_v2_smoke.py` | `pytest tests/test_omniview_fusion_v2_smoke.py -v` passes. | Test passes. |
| 0.3 | Confirm warm-start checkpoint availability (`ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_v2.pth` or 8.35 mm ensemble member). | `outputs/` manifest | File exists and loads. | Checkpoint loadable. |
| 0.4 | Build variable-view edge-index helper for graph-joint attention. | `motionflow_mv/fusion/graph_joint_utils.py` | Handles k=2..14 views without runtime error. | Edge index shape correct. |

### Phase 1 — GPU smoke (d=48, 10 epochs)

| # | Action | Artifact | Validation | Gate |
|---|--------|----------|------------|------|
| 1.1 | Run OmniMultiViewFusion v2 smoke on MPI-INF-3DHP S1 with `d=48`, 1 graph layer, visibility + uncertainty heads. | `outputs/omniview_v2_smoke_d48.log` | Clean MPJPE within **5%** of 9.32 mm baseline. | No crash; MPJPE ≤ 9.80 mm. |
| 1.2 | Ablate components: no visibility, no graph, isotropic uncertainty. | `experiments/prototypes/swarm_iter18/ablate_omniview_v2.py` | Per-ablation MPJPE and robustness delta. | Decide which heads justify their cost. |
| 1.3 | Generate first variable-view MPJPE@k curve. | `docs/figures/omniview_v2_variable_views_smoke.png` | k=14 within 0.5 mm of full-view; k=4 < 25 mm. | Curve non-degenerate. |

### Phase 2 — Full training (d=128, 20–30 epochs)

| # | Action | Artifact | Validation | Gate |
|---|--------|----------|------------|------|
| 2.1 | Full `d=128` training with warm-start and staged unfreezing. | `outputs/omniview_v2_d128.pth` | Clean MPJPE ≤ 8.0 mm; PA-MPJPE ≤ 5.0 mm. | Beats single-model anchor. |
| 2.2 | Extended robustness matrix: rot_0.5°, trans_5mm, focal_1%, pp_5/10px, joint occlusion 10/20/30%, view dropout 10/30/50%. | `outputs/robustness_omniview_v2.json` | All targets in §2 met. | Robustness story holds. |
| 2.3 | Repeated seeds (n ≥ 3) with `run_repeated_seeds.py`. | `outputs/repeated_seeds_omniview_v2/` | Mean clean MPJPE ≤ 8.0 mm; std ≤ 0.10 mm. | Reproducible. |

### Phase 3 — Paper package

| # | Action | Artifact | Validation | Gate |
|---|--------|----------|------------|------|
| 3.1 | Update `docs/icra_cvpr_2027_paper_story.md` with final numbers and fold in the new method subsection. | `docs/icra_cvpr_2027_paper_story.md` | Story consistent with experimental results. | Review pass. |
| 3.2 | Generate architecture figure and variable-view / robustness heatmaps. | `docs/figures/omniview_v2_*.png` | Figures match paper template. | Camera-ready quality. |
| 3.3 | Cross-dataset zero-shot check on H36M S9/S11. | `docs/results_omniview_v2_cross_dataset.md` | Diagnose H36M→MPI gap; do not block if >30 mm, but document. | Table complete. |

## 5. Open questions to resolve

1. **Uncertainty representation:** keep anisotropic 2×2 covariance (Bayesian tri v2) or switch to isotropic log-variance? Smoke both in Phase 1 and pick the one that improves clean MPJPE more.
2. **Graph-joint ordering:** before, after, or parallel to T×V attention? Run one ablation in Phase 1.2.
3. **Visibility loss weight:** sweep `λ_vis ∈ {0.05, 0.1, 0.2}` in Phase 1 to avoid collapse.
4. **Warm-start source:** use the 9.32 mm PP-corrected baseline or an 8.35 mm ensemble member? Prefer the stronger source if loadable.
5. **Cross-dataset strategy:** if H36M→MPI remains poor, queue a domain-adaptation wrapper *after* OmniMultiViewFusion is stable.

## 6. Risk register

| Risk | Mitigation | Owner artifact |
|------|------------|--------------|
| Unified model hurts clean accuracy | Warm-start + freeze encoder for 5 epochs; stage unfreezing. | trainer flags `--freeze_encoder_epochs` |
| Graph attention too slow / OOM | Use 1 graph layer only; benchmark on RTX 4090; optional gradient checkpointing. | `experiments/benchmark_runtime.py` |
| Visibility head collapses to all-ones | Fallback guard, BCE weight ≤ 0.1, synthetic occlusion labels. | `tests/test_visibility_collapse.py` |
| Uncertainty head explodes | Clamp log-variance to `[-5, 5]`. | model config |
| Variable-view edge rebuild breaks | Unit test edge index for k=2..14. | `tests/test_graph_joint_utils.py` |
| GPU queue bottleneck | No speculative runs; CPU work proceeds in parallel; smoke first. | this plan |

## 7. Immediate next steps (this week)

1. Verify `motionflow_mv/fusion/omniview_fusion_v2.py` compiles and passes the CPU smoke test in Phase 0.2.
2. If it does not exist yet, create the smoke test and run it.
3. Confirm the warm-start checkpoint path and add a trainer flag for staged freezing.
4. Queue the Phase 1 d=48 GPU smoke once CPU checks are green.

## 8. Definition of done for this synthesis

- [x] `docs/swarm_iter18/P20_synthesis.md` committed on `feat/swarm-iter18-omniview`.
- [ ] `tests/test_omniview_fusion_v2_smoke.py` passing (CPU).
- [ ] Phase 1 GPU smoke completed and logged.
- [ ] Decision log updated with answers to §5 open questions.

## 9. References

- `docs/swarm_iter18/P02_omniview_arch.md`
- `docs/swarm_iter18/P11_paper_story.md`
- `docs/design_omniview_fusion.md`
- `docs/iter_next_action_plan.md`
- `docs/results_icra_cvpr_2027.md`
- `docs/icra_cvpr_2027_paper_story.md`
