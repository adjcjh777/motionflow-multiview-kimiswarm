# Swarm Iter13 — Next-Iteration Synthesis (20-agent)

**Date:** 2026-08-06
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE
**Running GPU job:** `experiments/train_crossview_residual_visibility_v2_mpiinf3dhp.py` on RTX 4090
**SOTA context:** Classical DLT/robust IRLS baselines now land at **~25.2 mm** on the same split (`experiments/compare_sota_baselines.py`), giving the learned anchor a clean **15.9 mm** margin.
**Known blocker:** The PP-correction branch shows signs of saturation — the correction layer is driven only by downstream 3-D pose gradients and can under-/over-correct when `focal_max_scale` and PP noise are injected from epoch 1.

This document distills the latest 20-agent planning swarm into a minimal, executable roadmap. No new architecture is committed without a smoke test.

---

## 1. Current Empirical Anchor

- **Model:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
- **Metric:** 9.32 mm MPJPE / 5.37 mm PA-MPJPE on MPI-INF-3DHP clean
- **Headroom vs. SOTA triangulation:** DLT and robust IRLS are projected at ~25 mm on the same canonical split; the gap is large but must be reproduced under identical corruption conditions.
- **In-progress experiment:** `CrossviewResidualVisibilityV2` adds per-view/per-joint visibility gating plus a BCE loss. Result pending.
- **Calibration pain point:** PP/focal correction appears saturated. The correction module is trained indirectly and receives full camera noise from the first epoch, which limits both clean accuracy and robustness.

---

## 2. Thematic Synthesis of the 20 Proposals

### Group A — Visibility & Robustness (5 proposals)
Themes: explicit visibility gating, occlusion resilience, view/joint dropout, reprojection consistency, and 6-axis robustness matrices.
- The visibility v2 model is the active bet; several agents propose extending it with a visibility-conditioned residual, a reprojection-consistency loss, and richer camera augmentation (temporal consistency, radial distortion).
- A unified 6-axis robustness evaluator is the fastest way to prove whether these changes fix real failure modes.

### Group B — Architecture Efficiency & Inductive Bias (5 proposals)
Themes: factorized spatiotemporal attention, temporal smoothing, uncertainty/Gauss-Newton learned triangulation, skeleton-graph residual, and runtime compression.
- Factorized ST attention + PP correction is the cleanest near-term win if it preserves accuracy while cutting latency.
- Temporal smoothing and skeleton-aware residuals are higher-risk/higher-reward; both need a short smoke before committing.

### Group C — Data Scale, Mixing & Domain Adaptation (4 proposals)
Themes: WebBridge multi-dataset mixing, AIST++/Panoptic/3DPW conversion, SSL pre-training, and cross-dataset domain adaptation.
- The mixed loader exists but has no per-sample view/joint masks; mixed training is premature until masking is smoke-validated.
- SSL pre-training and domain adaptation are second-order experiments to queue only after visibility v2 and the factorized smoke land.

### Group D — Training & Calibration Refinement (3 proposals)
Themes: direct intrinsics-correction loss, camera-perturbation curriculum, and upgraded optimizer regime (AdamW + EMA + AMP + grad clip).
- The PP-correction saturation issue is best attacked first with a direct loss and a curriculum ramp, because it changes nothing in the model and could lift both clean and robust numbers.

### Group E — Evaluation, SOTA Alignment & Paper Assets (3 proposals)
Themes: SOTA comparison harness, unified benchmark protocol, failure analysis/diagnostics, and regenerated paper tables/figures.
- These are mostly CPU/A800-read-only tasks that can run in parallel with GPU training.
- They are also the highest paper-ROI tasks: they turn the 9.32 mm number into a defensible ICRA/CVPR 2027 claim.

---

## 3. Top-5 Prioritized Next Experiments

### 1. Complete visibility v2 training + occlusion evaluation (GPU)
**Files/scripts:**
- `experiments/train_crossview_residual_visibility_v2_mpiinf3dhp.py`
- `scripts/eval_crossview_residual_visibility_v2_wsl.sh`
- New: `experiments/eval_crossview_residual_visibility_v2_occlusion.py`

**Pass/fail:**
- Pass: clean val MPJPE ≤ 9.6 mm and PA-MPJPE ≤ 5.7 mm (within ~3% of anchor).
- Pass: at 30% view dropout, MPJPE is ≥ 10% lower than the PP baseline under the same dropout.
- Fail: clean MPJPE > 9.9 mm or occlusion gain < 5%.

**CPU/GPU:** GPU for training/eval; CPU for report generation.

**Self-evolution loop mapping:**
- *Reflect:* PP baseline is robust but not occlusion-aware; visibility gating is already training.
- *Hypothesize:* Visibility gating keeps clean accuracy while improving dropout robustness.
- *Smoke-validate:* Run the in-progress job to completion, then run clean + occlusion eval.
- *Integrate:* If it passes, make visibility v2 the new anchor and use it for robustness/SOTA comparisons.

---

### 2. Fix PP-correction saturation with direct intrinsics loss + curriculum (GPU)
**Files/scripts:**
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
- `scripts/run_crossview_pp_robust_retrain_wsl.sh`
- `motionflow_mv/fusion/principal_point_correction.py` (read-only; already returns `focal_scale`)
- `motionflow_mv/calibration/perturb.py` (for curriculum helpers)

**Pass/fail:**
- Pass: clean val MPJPE ≤ 9.7 mm on a 3–5 epoch smoke; no NaNs.
- Pass: Pearson correlation between predicted `pp_delta`/`focal_scale` and injected ground-truth perturbation > 0.3.
- Fail: MPJPE degrades > 0.4 mm or correlation ≤ 0.1.

**CPU/GPU:** GPU smoke on RTX 4090.

**Self-evolution loop mapping:**
- *Reflect:* PP/focal correction is learned indirectly and saturates because full noise is injected from epoch 1.
- *Hypothesize:* A direct intrinsics loss + curriculum ramp makes the correction more accurate and stable.
- *Smoke-validate:* Use the existing `--cam_aug_schedule intrinsics_curriculum` in the PP trainer and run a short smoke after visibility v2 finishes.
- *Integrate:* If it passes, fold into the visibility v2 full run and the factorized PP smoke.

---

### 3. Unified SOTA comparison harness (CPU / A800-D read-only)
**Files/scripts:**
- `experiments/compare_sota_baselines.py`
- `docs/tables/icra2027/main_results.md` / `.tex`
- `docs/paper_draft_icra_cvpr_2027.md`

**Pass/fail:**
- Pass: script completes in < 10 min on CPU for the clean test split.
- Pass: plain confidence-weighted DLT and robust IRLS produce finite, reproducible MPJPE (~25 mm expected).
- Pass: anchor checkpoint scores 9.32 mm on the exact same split.
- Fail: any method crashes, or anchor number deviates > 0.3 mm from 9.32 mm.

**CPU/GPU:** CPU; can read A800-D data read-only.

**Self-evolution loop mapping:**
- *Reflect:* No apples-to-apples comparison existed; SOTA numbers in the paper were stale or inferred.
- *Hypothesize:* A single harness will show the anchor’s margin over DLT/IRLS and expose where robust methods close the gap.
- *Smoke-validate:* Run the harness on the clean test split and verify DLT/IRLS/anchor numbers.
- *Integrate:* Regenerate paper tables and archive JSON in `docs/tables/icra2027/sota_comparison_mpi.json`.

---

### 4. 6-axis robustness matrix on the PP baseline and visibility v2 (CPU / GPU eval)
**Files/scripts:**
- `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`
- `experiments/plot_robustness_matrix.py`
- `motionflow_mv/eval/benchmark_protocol.py`
- `outputs/robustness_matrix_pp_full.json`

**Pass/fail:**
- Pass: clean MPJPE ≤ 9.6 mm on the matrix smoke.
- Pass: no moderate-severity condition degrades > 30% relative to clean.
- Pass: runtime < 10 min smoke; JSON + Markdown outputs are produced.
- Fail: missing sections, clean deviation > 0.5 mm, or severe regressions on any axis.

**CPU/GPU:** CPU for analysis/eval; GPU only if learned-model inference is required.

**Self-evolution loop mapping:**
- *Reflect:* Robustness claims are currently anecdotal; a matrix exists but only for the PP baseline at fixed severity.
- *Hypothesize:* Severity sweeps + combined corruptions will reveal which axes visibility gating actually helps.
- *Smoke-validate:* Add severity sweeps and a visibility-v2 path, run on a small subset.
- *Integrate:* Promote to the standard eval for every future model checkpoint.

---

### 5. Factorized ST attention + PP correction smoke (GPU)
**Files/scripts:**
- New: `motionflow_mv/fusion/ray_attention_temporal_crossview_factorized_residual_principal_point_model.py`
- New: `experiments/train_factorized_pp_smoke_mpiinf3dhp.py`
- `experiments/benchmark_runtime.py` (add new model)

**Pass/fail:**
- Pass: clean val MPJPE ≤ 9.8 mm (≤ 0.5 mm above the 9.32 mm baseline).
- Pass: single-frame and single-clip latency reduced by ≥ 30% on the same GPU.
- Pass: no NaNs/crashes.
- Fail: accuracy drop > 0.5 mm or speedup < 30%.

**CPU/GPU:** GPU smoke on RTX 4090 or A800-D.

**Self-evolution loop mapping:**
- *Reflect:* The anchor is accurate but uses expensive `(time × view)` attention; a factorized variant already exists but lacks PP correction.
- *Hypothesize:* Adding PP correction to the factorized model preserves accuracy while cutting latency.
- *Smoke-validate:* Create the PP factorized model, train 5 epochs, benchmark latency.
- *Integrate:* If it passes, replace the anchor with the faster model for production and future ablations.

---

## 4. GPU Queue Recommendation Order

Only one RTX 4090 is available and it is currently training visibility v2. CPU/read-only tasks can run in parallel.

1. **RTX 4090 — Finish visibility v2 training** (in progress).
   - Blocker: nothing else can run here until it finishes.
2. **RTX 4090 — PP-correction saturation fix smoke** (Experiment #2).
   - Short 3–5 epoch run; if it passes, fold into the next full visibility v2 re-run.
3. **A800-D GPUs 0–3 (after owner confirmation) — Factorized ST+PP smoke** (Experiment #5).
   - Independent of visibility v2; can run in parallel on A800-D if policy allows.
4. **RTX 4090 / A800-D — Full visibility v2 re-train with PP fix if Experiment #2 passes**.
5. **GPU eval only** for robustness matrix and occlusion eval can reuse the trained checkpoints without training.

Parallel CPU/A800-read-only work (no GPU needed):
- Implement & run `compare_sota_baselines.py` (Experiment #3).
- Extend and run `eval_robustness_matrix_pp_mpiinf3dhp.py` (Experiment #4).
- Regenerate paper tables/figures.
- Failure analysis on visibility v2 checkpoint.

---

## 5. Immediate Action Items

- [ ] Monitor `train_crossview_residual_visibility_v2_mpiinf3dhp.py` to completion and run `scripts/eval_crossview_residual_visibility_v2_wsl.sh`.
- [x] Add direct PP/focal intrinsics loss and curriculum ramp to the PP trainer; queued via `scripts/run_crossview_pp_robust_retrain_wsl.sh`.
- [x] Run `experiments/compare_sota_baselines.py` and update paper tables (DLT/IRLS/anchor comparison done).
- [ ] Extend `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` with severity sweeps and a visibility-v2 path; smoke on 20 clips.
- [ ] Create the factorized ST+PP model + smoke trainer; benchmark latency vs. the 9.32 mm baseline.
- [x] Regenerate `docs/tables/icra2027/main_results.md` and paper draft from the 9.32 mm anchor.
- [ ] Confirm A800-D GPU policy/availability with the owner before scheduling Experiment #5 there.

---

## 6. Qwen-Style Self-Evolution Loop

Every top experiment follows **reflect → hypothesize → smoke-validate → integrate**:

| Experiment | Reflect | Hypothesize | Smoke-validate | Integrate |
|---|---|---|---|---|
| 1. Visibility v2 + occlusion eval | PP baseline is occlusion-naïve | Visibility gating improves dropout robustness at no clean cost | Finish training; run clean + 30% dropout eval | Make v2 the anchor if it passes |
| 2. PP saturation fix | PP/focal correction saturates under indirect loss | Direct intrinsics loss + curriculum improves correction | 3–5 epoch smoke with correlation check | Apply to v2 and factorized runs |
| 3. SOTA comparison harness | No apples-to-apples SOTA numbers | DLT/IRLS ~25 mm; anchor holds 9.32 mm | Build harness on small split, compare | Feed into paper tables and unified benchmark |
| 4. 6-axis robustness matrix | Robustness claims are anecdotal | Severity sweeps show real v2 gains | Smoke 20 clips with severity/combined axes | Standard eval for every checkpoint |
| 5. Factorized ST+PP | Anchor attention is expensive | Factorization + PP keeps accuracy, cuts latency | 5-epoch train + latency benchmark | Replace production anchor if speedup ≥ 30% |
