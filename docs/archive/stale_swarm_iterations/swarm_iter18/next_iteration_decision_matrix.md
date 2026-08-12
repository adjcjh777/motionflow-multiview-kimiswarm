# Next-Iteration Decision Matrix — OmniMultiViewFusion v2

**Date:** 2026-08-07  
**Status:** Living decision matrix for swarm-iter18/iter19  
**Scope:** Decide how to proceed after the OmniMultiViewFusion v2 full run, based on clean accuracy, robustness, variable-view behaviour, and training stability.  
**Anchor to beat:** Bayesian Tri v2 ensemble at **8.35 mm** MPJPE / **5.29 mm** PA-MPJPE on MPI-INF-3DHP S2/Seq1.  
**Single-model target:** ≤ **8.0 mm** MPJPE, ≤ **5.0 mm** PA-MPJPE.

---

## 1. Decision axes

Evaluate the v2 full-run outcome along four axes.  All four must be reported before a decision is declared.

| # | Axis | Buckets / gates |
|---|---|---|
| **A** | **Clean MPJPE (MPI-INF-3DHP S2/Seq1)** | **< 8.0 mm** (target) · **8.0–8.35 mm** (near target) · **8.35–9.0 mm** (below old anchor, not enough) · **> 9.0 mm** (regression) |
| **B** | **Calibration robustness** | **Pass:** rot_0.5° < 12 mm, focal_1% < 14 mm, pp_10px < 50 mm, no catastrophic jump on trans_5/10 mm.  **Fail:** any gate exceeded or clean accuracy collapses under perturbation. |
| **C** | **Variable-view k = 4** | **Pass:** MPJPE@k=4 < 60 mm.  **Fail:** ≥ 60 mm or non-monotonic / noisy curve. |
| **D** | **Training stability** | **Pass:** no NaNs, no divergence, epoch time within 1.3× of Bayesian Tri v2 baseline, memory < 24 GB peak.  **Fail:** NaNs, loss explosion, >1.5× epoch time, OOM. |

> **Note on variable-view target.**  The aspirational target in `P20_synthesis.md` is k=4 < 25 mm, but the current Bayesian Tri v2 stabilized checkpoint shows k=4 ≈ 113 mm (`docs/results_icra_cvpr_2027.md`).  The 60 mm gate is chosen as a realistic, non-trivial improvement that proves the visibility/graph mechanism is working; it will be tightened once this gate is consistently met.

---

## 2. Outcome matrix

The matrix is ordered from most favourable to least favourable.  For each cell, the action is conditional on the **combination** of axes, not any single axis in isolation.

### 2.1 Decision tree summary

```
Clean MPJPE < 8.0 mm  ──►  ADOPT as new anchor (if robustness + stability pass)
          │
          └─ 8.0–8.35 mm  ──►  CONDITIONAL ADOPT or MINOR ITERATE
          │
          └─ 8.35–9.0 mm  ──►  ITERATE (diagnose component, one-variable-at-a-time)
          │
          └─ > 9.0 mm      ──►  ABANDON or HEAVY FALLBACK
```

### 2.2 Detailed combinations

#### Row 1: Clean < 8.0 mm (target exceeded)

| Robustness | k=4 < 60 mm | Stability | Action | Next experiments | Keep / merge / remove | Timeline / GPU |
|---|---|---|---|---|---|---|
| Pass | Yes | Pass | **ADOPT as new anchor** | Run repeated seeds (n ≥ 3); full extended robustness matrix; generate variable-view curve and paper figures; cross-dataset zero-shot on H36M S9/S11. | **Keep:** full v2 module (intrinsic self-calibration, visibility, graph-joint attention, T×V×J attention, uncertainty-weighted triangulation, Gauss-Newton refinement, residual correction). **Merge** into `MultiViewFusionPlugin` interface. **Remove:** old standalone ensemble scripts from default path; keep as legacy plugin. | 1–2 weeks: 3–5 seed runs + robustness matrix (~300 RTX 4090 h). 1 week: figures, paper story update. |
| Pass | Yes | Fail | **ITERATE on stability** | Roll back to last stable checkpoint; isolate the divergence component (likely T×V×J attention or adaptive GN step size). Reduce learning rate, add gradient clipping, or replace dense attention with factorised form. | **Keep:** encoder + visibility + uncertainty heads. **Temporarily remove or freeze:** graph-joint attention and GN refinement until stability proven. | 2–3 days diagnostic run; 5–7 days re-training. |
| Pass | No | Pass | **CONDITIONAL ADOPT + variable-view sprint** | Visibility head is probably under-trained for low view counts. Add synthetic view dropout ≥ 30 % during training; sweep visibility loss weight λ_vis ∈ {0.05, 0.1, 0.2}; run variable-view edge-index ablation. | **Keep:** full module. **Add:** dedicated low-view visibility loss and data augmentation. | 1 week ablation; 1 week re-train if promising. |
| Fail | Yes | Pass | **MINOR ITERATE** | Robustness failure means calibration components need tuning. Queue calibration curriculum v2 (P0.2 in `docs/iter_next_action_plan.md`) on top of the v2 checkpoint. | **Keep:** full module. **Add:** staged perturbation curriculum; bound intrinsic corrections to dataset ranges. | 1 week curriculum tuning; 1 week re-train. |
| Fail | No | — | **ITERATE heavily** | The unified architecture is not yet stable. Diagnose whether robustness or variable-view is the primary failure, then address the dominant one before declaring anchor. | **Keep:** geometry core (intrinsic self-calibration + uncertainty triangulation). **Remove or freeze:** graph-joint and T×V×J attention until baseline robustness recovers. | 2 weeks targeted ablations; 1 week re-train. |

#### Row 2: Clean 8.0–8.35 mm (near target, matches or slightly above ensemble)

| Robustness | k=4 < 60 mm | Stability | Action | Next experiments | Keep / merge / remove | Timeline / GPU |
|---|---|---|---|---|---|---|
| Pass | Yes | Pass | **CONDITIONAL ADOPT (iterative polish)** | A single model at 8.0–8.35 mm is publication-ready but not a clear win over the 8.35 mm ensemble. Run: (1) 3–5 seeds, (2) calibration curriculum on top, (3) small capacity sweep (d=96 vs. d=128). If any seed reaches < 8.0 mm, declare new anchor. | **Keep:** full v2 module. **Merge:** only if repeated-seed mean < 8.35 mm. | 1 week seeds; 1 week curriculum. |
| Pass | Yes | Fail | **ITERATE on stability** | Same as Row 1 stability-fail. | **Keep:** majority of module. **Remove/freeze** unstable head. | 3–5 days. |
| Pass | No | Pass | **ITERATE on variable-view** | Clean accuracy is good but low-view behaviour is poor. Synthetic view dropout and visibility loss sweep are the priority. | **Keep:** full module. **Add:** low-view augmentation. | 1 week. |
| Fail | Yes | Pass | **MINOR ITERATE (robustness)** | Run calibration curriculum v2, bound corrections, and add rotation-aware correction head (risk register in `docs/paper_story_system_v2.md`). | **Keep:** full module. **Add:** curriculum. | 1 week. |
| Fail | No | — | **ABANDON unified direction (temporarily)** | Fallback to Bayesian Tri v2 + incremental improvements (see §3.1). | **Keep:** intrinsic self-calibration and uncertainty heads as isolated plugins. **Remove:** integrated graph/T×V×J attention from default branch. | 2 weeks fallback build. |

#### Row 3: Clean 8.35–9.0 mm (below old anchor, not enough)

| Robustness | k=4 < 60 mm | Stability | Action | Next experiments | Keep / merge / remove | Timeline / GPU |
|---|---|---|---|---|---|---|
| Pass | Yes | Pass | **ITERATE one component at a time** | The architecture is viable but under-trained or under-sized. Queue: (1) capacity sweep d=128 → d=192/d=256, (2) longer training (40–50 epochs), (3) better warm-start from 8.35 mm ensemble member, (4) ablate graph-joint ordering. | **Keep:** full module. **Tweak:** capacity and training length; no removals. | 1–2 weeks per ablation; 2–3 weeks total. |
| Pass | Yes | Fail | **ITERATE on stability + capacity** | Stabilise first, then re-run capacity sweep. | **Keep:** encoder + visibility + uncertainty. **Freeze/remove:** graph and temporal attention until stable. | 2 weeks. |
| Pass | No | Pass | **ITERATE on variable-view** | Robustness is fine; low-view is the bottleneck. Synthetic view dropout and low-view loss are the first experiments. | **Keep:** full module. **Add:** view-dropout augmentation + visibility loss tuning. | 1–2 weeks. |
| Fail | Yes | Pass | **ITERATE on calibration robustness** | Apply curriculum v2 and bound intrinsic corrections. If robustness improves without hurting clean accuracy, proceed to capacity sweep. | **Keep:** full module. **Add:** curriculum. | 1–2 weeks. |
| Fail | No | — | **ABANDON unified direction** | Neither clean accuracy nor robustness/variable-view justify continuing. Fallback to Bayesian Tri v2 + incremental improvements. | **Keep:** proven components as plugins. **Remove:** integrated v2 from default. | 2 weeks fallback. |

#### Row 4: Clean > 9.0 mm (regression vs. ensemble)

| Robustness | k=4 < 60 mm | Stability | Action | Next experiments | Keep / merge / remove | Timeline / GPU |
|---|---|---|---|---|---|---|
| Pass | Yes | Pass | **ITERATE heavily / conditional salvage** | Regression is likely due to a single head (visibility collapsing, graph attention over-smoothing, or GN step hurting clean accuracy). Run isolation ablations: no visibility, no graph, isotropic uncertainty, 0 GN steps. If any isolated run ≥ 8.35 mm, that component is the culprit. | **Keep:** proven geometry core. **Temporarily remove:** suspected culprit head. | 1 week ablations; 1 week re-train. |
| Pass | Yes | Fail | **ABANDON or SALVAGE** | Fix stability first. If still > 9.0 mm after stability fix, abandon. | **Remove:** unstable components. | 1 week. |
| Pass | No | Pass | **ABANDON unified; salvage components** | The module is not deployment-ready. Extract visibility and intrinsic correction as separate plugins. | **Keep:** visibility head and PP correction as plugins. **Remove:** integrated v2. | 2 weeks. |
| Fail | Yes | Pass | **ABANDON; heavy fallback** | Robustness failure on a > 9.0 mm model is not worth salvaging. | **Keep:** only triangulation + uncertainty. **Remove:** graph, T×V×J attention, GN refinement from default. | 2 weeks fallback. |
| Fail | No | — | **ABANDON direction** | All axes fail. Full fallback to Bayesian Tri v2 + incremental improvements. | **Remove / archive:** integrated v2 into a prototype branch. **Keep:** interface contract unchanged. | 2 weeks fallback. |

---

## 3. Contingency plans

### 3.1 If OmniMultiViewFusion underperforms (< 8.35 mm or fails robustness gates)

The fallback is **not** to start from scratch.  It is a staged retreat to the proven Bayesian Tri v2 backbone plus the smallest set of validated improvements.

| Step | Fallback action | Target | Validation |
|---|---|---|---|
| F1 | Re-confirm Bayesian Tri v2 single-model checkpoint (`outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth`) and ensemble script. | Reproduce ≤ 9.03 mm single, ≤ 8.35 mm ensemble. | `scripts/eval_ensemble_wsl.sh` |
| F2 | Add **intrinsic self-calibration head only** to Bayesian Tri v2. | Clean ≤ 8.8 mm, rot_0.5° < 14 mm. | Robustness matrix |
| F3 | Add **uncertainty-weighted triangulation** (isotropic log-variance). | Clean ≤ 8.7 mm, focal_1% < 14 mm. | Robustness matrix |
| F4 | Add **visibility head** only if it passes isolation smoke. | ≥ 10 % relative gain at 30 % occlusion, clean ≤ 8.75 mm. | Occlusion sweep |
| F5 | Add **calibration curriculum v2** (P0.2). | rot_0.5° < 12 mm, focal_1% < 14 mm. | Extended robustness matrix |

**Fallback stop condition:** Any step that fails its gate is rolled back.  The direction is abandoned when the stack reaches ≥ 8.75 mm single-model clean MPJPE *and* robustness does not improve over the 8.35 mm ensemble.

### 3.2 If OmniMultiViewFusion overperforms (< 8.0 mm with clean robustness)

| Stack item | What | Rationale | Success gate | Timeline / GPU |
|---|---|---|---|---|
| S1 | **SSL masked-view pre-training** (P1.3 in `docs/iter_next_action_plan.md`). | Data-efficiency narrative: pre-train on unlabeled multi-view clips, fine-tune on MPI. | ≥ 10 % data reduction at same MPJPE. | 2–3 weeks pre-train + fine-tune. |
| S2 | **Cross-dataset training (MPI + H36M + AIST)** (P1.2, P18 cross-dataset plan). | Improve generalisation and paper tables. | H36M S9/S11 MPJPE < 30 mm; MPI clean maintained < 8.0 mm. | 2–3 weeks mixed training. |
| S3 | **Larger capacity (d=192/d=256) + longer schedule**. | Push clean accuracy floor. | Clean ≤ 7.5 mm single model. | 1–2 weeks. |
| S4 | **Robot-profile zero regression + end-to-end policy validation**. | Required for ICRA/CVPR 2027 system claim. | `bxi_elf3_current` profile matches hard-coded pipeline; policy metrics collected. | 2–4 weeks. |
| S5 | **Real-time inference optimization** (P19). | Throughftargets for paper. | ≥ 100 clips/s on RTX 4090, MPJPE < 8.0 mm. | 1 week. |

**Stacking order:** S1 and S3 can run in parallel after adoption.  S2 depends on S1 only if SSL pre-training improves cross-dataset transfer.  S4 is the system-level capstone and is queued after S2/S3.

---

## 4. Stop conditions for the current direction

Stop continuing to invest in the unified OmniMultiViewFusion v2 direction when **any** of the following are true:

1. **Clean accuracy ceiling:** after three independent full runs (different seeds or minor hyperparameter tweaks), the best clean MPJPE is still ≥ 8.35 mm and no run passes robustness gates.
2. **Stability failure:** two consecutive full runs hit NaNs/divergence despite gradient clipping, learning-rate reduction, and component freezing.
3. **Robustness regression:** the unified model is worse than Bayesian Tri v2 on **two or more** of rot_0.5°, focal_1%, pp_10px after a curriculum attempt.
4. **Variable-view regression:** k=4 MPJPE is not improving after synthetic view-dropout training and visibility loss tuning.
5. **Epoch-time blowout:** v2 epoch time > 1.5× Bayesian Tri v2 baseline and cannot be reduced by gradient checkpointing or smaller batch size without accuracy loss.
6. **Opportunity cost:** the fallback stack (§3.1) reaches < 8.75 mm single-model with better robustness in less total GPU time.

---

## 5. Criteria for declaring a new anchor

A model becomes the new anchor only when **all** of the following are satisfied:

| # | Criterion | Threshold | Evidence |
|---|---|---|---|
| 1 | Clean MPJPE | **< 8.35 mm** (single model) | Validation run on full MPI-INF-3DHP S2/Seq1 with `val_stride=50` |
| 2 | PA-MPJPE | **≤ 5.0 mm** | Same validation run |
| 3 | Repeated seeds | **n ≥ 3**, mean clean MPJPE lower than current anchor, std ≤ 0.10 mm | `experiments/run_repeated_seeds.py` output |
| 4 | Robustness matrix | rot_0.5° < 12 mm, focal_1% < 14 mm, pp_10px < 50 mm, trans_5/10 mm < 12 mm | `experiments/prototypes/run_extended_robustness_matrix.py` |
| 5 | Variable-view curve | k=14 within 0.5 mm of full-view; k=4 < 60 mm (tighten to < 25 mm once gate is met) | `docs/figures/omniview_v2_variable_views.png` |
| 6 | Training stability | No NaNs, no divergence, peak memory < 24 GB, epoch time < 1.3× baseline | Training log / `manifest.json` |
| 7 | Reproducibility | Checkpoint, log, config, and `manifest.json` committed | Git commit hash recorded |
| 8 | Downstream smoke | `HumanMotionIR` compatibility export passes golden-artifact regression | `tests/test_humanmotionir_export.py` |

Once all criteria are met:

1. Update `docs/results_icra_cvpr_2027.md` with the new numbers.
2. Update `docs/paper_story_system_v2.md` §4.2 with the new anchor and checkpoint path.
3. Rename/move the winning checkpoint to `outputs/omniview_v2_anchor_mpiinf3dhp.pth`.
4. Update `docs/iter_next_action_plan.md` §1 “Current state” and re-prioritise the remaining P0/P1/P2 items.
5. Generate the paper-package figures (architecture, variable-view curve, robustness heatmap) and queue the system-level experiments (§3.2).

---

## 6. GPU allocation policy

| Phase | Typical GPU need | Allocation rule |
|---|---|---|
| Diagnostic / ablation (d=48, 10–20 epochs) | 1× RTX 4090, 12–24 h | One slot until gate is decided. |
| Full d=128 run | 1× RTX 4090, 5–7 days | Only one full run at a time; no speculative parallel full runs. |
| Repeated seeds | 1× RTX 4090, 2–3 days | Sequential or parallel depending on queue availability; n ≥ 3 required. |
| Robustness matrix | 1× RTX 4090, 1 day | Run only after clean accuracy gate passed. |
| Cross-dataset / SSL | 1× RTX 4090, 1–3 weeks | Queued after anchor declared. |

**Hard rule:** CPU-only work (documentation, test hardening, figure generation, tracker validation) proceeds in parallel; no new GPU training is queued until the current full run has been evaluated against this matrix.

---

## 7. Related documents

- `docs/paper_story_system_v2.md` — system narrative and risk register.
- `docs/swarm_iter18/P20_synthesis.md` — P0/P1/P2 action plan and open questions.
- `docs/iter_next_action_plan.md` — gated next-iteration plan and gating rules.
- `docs/results_icra_cvpr_2027.md` — current experimental results.
- `docs/icra_cvpr_2027_paper_story.md` — paper story and figures plan.
