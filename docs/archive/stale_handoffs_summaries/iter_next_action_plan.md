# Iteration Next Action Plan — MotionFlow-MultiView

**Date:** 2026-08-06  
**Goal:** Reach ICRA / CVPR 2027 publishable quality on MPI-INF-3DHP, near-term target **MPJPE < 8.75 mm** on the validation set.  
**Anchor experiment:** `scripts/run_bayesian_tri_v2_large_scale_wsl.sh` (d=128, residual_hidden=256, n_st_layers=3, 50 epochs, RTX 4090).  
**Anchor log:** `outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.log`.  

This document is a synthesis of the 20-agent swarm outputs in `docs/swarm_iter_next/` and `docs/next_iteration_plan_swarm.md`. It turns the 20 exploration directions into a single ranked, gated action plan.

---

## 1. Current state

| Item | Value |
|------|-------|
| Current best clean MPJPE | **8.75 mm** / PA-MPJPE **4.95 mm** (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth`) |
| Running anchor | Bayesian Tri v2 large scale — currently ~epoch 13, best so far **9.81 mm** (did not beat anchor yet) |
| GPU policy | **No new GPU training until the anchor run finishes.** CPU-only work, smoke tests, and documentation proceed in parallel. |
| Main weakness | Calibration robustness: rot_0.5° → 16.89 mm, focal_1% → 19.13 mm, principal-point ±10 px catastrophic. |
| Secondary wins | Variable-view harness, WebBridge benchmark, repeated-seed runner, SSL skeleton, failure-analysis scripts landed in iter 2026-08-06. |

---

## 2. Guiding principles

1. **One variable at a time.** Each new experiment must isolate a single change and compare against the 8.75 mm anchor.  
2. **Fail fast on CPU.** Every new module needs a CPU smoke test before it is allowed to queue for GPU.  
3. **No speculative stacking.** Do not combine visibility gating, spatiotemporal Transformer, and calibration curriculum in one run until each is individually validated.  
4. **Publishable evidence first.** Any change that reaches < 8.75 mm must also produce a robustness matrix, a variable-view curve, and a repeated-seed result before it is declared the new anchor.  

---

## 3. Prioritized action plan

### 3.1 P0 — Unlock the next anchor (do now / queue first)

| # | Action | Rationale | Owner artifact | Validation | Success gate |
|---|--------|-----------|----------------|------------|--------------|
| P0.1 | **Finish & evaluate Bayesian Tri v2 large scale** | Already running; largest capacity attempt. | `scripts/run_bayesian_tri_v2_large_scale_wsl.sh` | Clean MPJPE + PA-MPJPE + robustness matrix | If MPJPE < 8.75 mm, run 3–5 seeds and declare new anchor. |
| P0.2 | **Calibration robustness curriculum v2** | The single biggest regression under perturbation. | `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` | `eval_perturb_model_mpiinf3dhp.py` matrix | clean ≤ 9.6 mm, rot_0.5° < 12 mm, focal_1% < 14 mm. |
| P0.3 | **Visibility-gated fusion v2** | Occlusion is the deployment gap; explicit visibility head is wired and CPU-smoked. | `motionflow_mv/fusion/visibility_gated_fusion.py`, `experiments/train_crossview_residual_visibility_v2_mpiinf3dhp.py` | Clean + occlusion sweep (0/10/30/50%) | clean ≤ 9.6 mm, ≥ 10% relative gain at 30% occlusion. |
| P0.4 | **Variable-view inference + view-dropout training** | Practical rigs have 2–10 views, not 14. Harness exists but not trained in. | `experiments/eval_variable_views.py`, `experiments/plot_variable_views.py` | MPJPE@k for k=2..14 on anchor checkpoint | graceful degradation; k=14 matches 8.75 mm. |
| P0.5 | **Unified benchmark protocol & repeated seeds** | Required for any publishable claim. | `motionflow_mv/eval/benchmark_protocol.py` (planned), `experiments/run_repeated_seeds.py` | 3–5 seeds, manifest JSON per run | mean ± std reported for every anchor candidate. |

**Execution order:**

- **Now (CPU):** P0.4 variable-view curve on anchor; P0.5 benchmark protocol skeleton; P0.2 design review and hyperparameter grid.
- **Next GPU slot after Bayesian Tri v2:** P0.2 calibration curriculum v2 if Bayesian Tri v2 did not beat 8.75 mm; otherwise evaluate P0.2 on the new anchor.
- **GPU slot 2:** P0.3 visibility-gated v2.
- **GPU slot 3:** P0.2 + P0.3 combined only if both individually pass gates.

### 3.2 P1 — Consolidate the story (queue after P0 gates)

| # | Action | Rationale | Owner artifact | Validation | Success gate |
|---|--------|-----------|----------------|------------|--------------|
| P1.1 | **Spatiotemporal (T × V × J) Transformer** | Potential clean MPJPE < 9 mm; expensive, so only after P0. | `motionflow_mv/fusion/ray_attention_spatiotemporal_model.py` | CPU smoke + 20-epoch small run | clean < 8.75 mm or ≥ 0.3 mm improvement over anchor. |
| P1.2 | **Cross-dataset WebBridge benchmark** | Paper needs MPI / H36M / AIST / Shelf / Campus tables. | `experiments/run_webbridge_benchmark.py` | Per-dataset MPJPE/PA table | Diagnose and fix H36M 101 mm regression. |
| P1.3 | **Self-supervised masked-view pre-training** | Data-efficiency narrative; low risk if kept additive. | `motionflow_mv/data/ssl_dataset.py`, `experiments/pretrain_ray_attention_ssl.py` | Pre-train → fine-tune curve vs. supervised baseline | ≥ 10% data reduction at same MPJPE. |
| P1.4 | **Uncertainty-aware per-view weighting** | Interpretable confidence fusion; complements P0.3. | `motionflow_mv/fusion/ray_attention_temporal_uncertainty_model.py` | Smoke + small ablation | clean ≤ 9.0 mm or visibly improves robustness matrix. |
| P1.5 | **Temporal consistency / longer clips** | Reduce jitter; useful if velocity metric is poor. | `motionflow_mv/losses/temporal_consistency.py` (in progress) | velocity MPJPE on 25-frame clips | velocity MPJPE reduced ≥ 5%. |
| P1.6 | **Graph joint relation v2** | Helps H36M; may not help MPI. Revisit after cross-dataset fix. | `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_graph_joint_model.py` | H36M S5/Act2 MPJPE | H36M < 0.8 mm. |

### 3.3 P2 — Stretch / system-level (ICRA/CVPR 2027 late stage)

| # | Action | Rationale | Owner artifact | Validation | Success gate |
|---|--------|-----------|----------------|------------|--------------|
| P2.1 | **Real-time inference optimization** | Latency/throughput numbers for paper. | `experiments/benchmark_runtime.py` | FPS on RTX 4090, batch=1 | ≥ 30 FPS with MPJPE < 9 mm. |
| P2.2 | **Multi-person multi-view association** | System extension; new application scenario. | `experiments/associate_multi_person_synthetic.py` | Synthetic 2-person clip | IDF1 > 0.90. |
| P2.3 | **Action-conditional fusion** | Per-action error reduction on H36M. | `experiments/ablate_action_aware.py` | Per-action MPJPE | ≥ 2 joints improved on worst action class. |
| P2.4 | **Gaussian splatting pose regularizer** | Novel auxiliary signal; risky. | `experiments/test_gaussian_splatting_pose_loss.py` | Isolated smoke | No catastrophic regression on 2-epoch smoke. |

---

## 4. Phase schedule

### Phase 0 — Non-GPU preparation (now)

- [ ] Produce full MPJPE@k curve (k=2..14) for the 8.75 mm anchor.
- [ ] Harden `motionflow_mv/eval/benchmark_protocol.py` and wire it to `run_repeated_seeds.py`.
- [ ] Audit WebBridge `.npz` quality and produce a cross-dataset baseline table (CPU-only subset).
- [ ] Finalize P0.2 curriculum hyperparameters (rot/trans/focal/pp schedule).
- [ ] Ensure P0.3 visibility-gated v2 trainer is GPU-ready and documented.

**Stop condition for Phase 0:** All P0 CPU deliverables committed and at least one GPU slot freed.

### Phase 1 — GPU convergence (after Bayesian Tri v2 finishes)

- [ ] Evaluate Bayesian Tri v2 final checkpoint.
- [ ] If MPJPE ≥ 8.75 mm, run P0.2 calibration curriculum v2.
- [ ] Run P0.3 visibility-gated v2.
- [ ] If P0.2 and P0.3 both pass gates, run a combined curriculum + visibility experiment.
- [ ] Update anchor and run repeated seeds.

**Stop condition for Phase 1:** A model reaches **MPJPE < 8.75 mm** on clean validation and passes the robustness matrix.

### Phase 2 — Paper package (after Phase 1 anchor is set)

- [ ] Run P1.2 WebBridge benchmark and fix H36M 101 mm issue.
- [ ] Run P1.3 SSL pre-training data-efficiency curve.
- [ ] Add P1.1 spatiotemporal Transformer if compute allows.
- [ ] Generate all paper tables/figures and update `docs/paper_draft_icra_cvpr_2027.md`.

---

## 5. Gating rules

A candidate model may replace the current anchor only if:

1. Clean MPI-INF-3DHP S2/Seq1 MPJPE < 8.75 mm (or current anchor, whichever is lower).
2. Robustness matrix: rot_0.5° < 12 mm, focal_1% < 14 mm, pp_10px < 50 mm.
3. Repeated seeds (n ≥ 3): mean clean MPJPE is lower than current anchor with no single seed > 10% worse.
4. Variable-view MPJPE@k curve is non-degenerate (k=14 within 0.5 mm of full-view, k=4 < 20 mm).
5. Checkpoint and training log committed; `manifest.json` generated.

---

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Bayesian Tri v2 does not beat 8.75 mm | Medium | P0.2 and P0.3 are already queued as follow-ups. |
| Calibration curriculum overfits to perturbation | Medium | Bound perturbation magnitude; keep a clean-only validation track. |
| Visibility head collapses to all-occluded | Low | Fallback guard + BCE weight ≤ 0.1 + warm-start from anchor. |
| H36M 101 mm regression blocks cross-dataset table | High | Debug in `run_webbridge_benchmark.py` before declaring cross-dataset results. |
| RTX 4090 bottleneck | High | Keep CPU work parallel; queue GPU experiments by priority; no speculative runs. |

---

## 7. Definition of done for this synthesis task

- [ ] This plan (`docs/iter_next_action_plan.md`) is committed on `feat/iter-next-synthesize-swarm-outputs`.
- [ ] A machine-readable tracker (`experiments/prototypes/iter_next_action_tracker.py`) can load and validate the plan items.
- [ ] A CPU smoke test (`tests/test_iter_next_action_tracker.py`) passes.
- [ ] Branch is pushed to `origin`.

---

## 8. References

- `docs/next_iteration_plan_swarm.md` — 20-agent swarm synthesis with 20 directions.
- `docs/swarm_iter_20260806_summary.md` — latest iteration summary (best 9.32 / 5.37 mm).
- `docs/results_iter16.md` — iter16 results including Bayesian Tri v2 partial log.
- `docs/experiment_log_icra_cvpr_2027.md` — chronological experiment log.
- `docs/swarm_iter_next/design_*` — per-direction design documents.
