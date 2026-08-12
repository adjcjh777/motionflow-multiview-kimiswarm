# Ensemble of Factorized and Temporal Anchors for Lower Error

**Date:** 2026-08-06  
**Author:** MotionFlow-MultiView planning swarm (iter14)  
**Status:** Proposal — ready for smoke test  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE

---

## 1. Problem

The current 9.32 mm anchor relies on expensive `(time × view)` attention, while the faster factorized variant and the occlusion-aware visibility-v2 branch reach only intermediate states; no mechanism exploits the complementary strengths of these independently trained anchors to push the final error lower.

---

## 2. Hypothesis

A lightweight, late-stage ensemble that fuses predictions from the strong temporal anchor, the factorized ST anchor, and (when available) the visibility-v2 anchor—using per-joint learned weights or a simple residual-correcting meta-head—can reduce clean MPJPE by 0.3–0.8 mm with negligible runtime cost after the constituent models are loaded.

---

## 3. Method

### 3.1 Architecture / Data / Loss Changes

1. **New ensemble inference module**  
   - Create `motionflow_mv/fusion/ensemble_anchor_model.py`  
     - Input: list of 3-D pose predictions `(B, T, J, 3)` and, optionally, per-model confidence scores `(B, T, J)` or per-model latent vectors.  
     - Two variants:
       - `AverageEnsemble`: weighted average with scalar weights learned per joint.
       - `ResidualEnsemble`: a tiny 2-layer MLP (input = concatenated predictions + confidences, hidden = 64, output = residual offset added to the weighted average).
     - Output: fused 3-D pose `(B, T, J, 3)`.
   - The module is **late fusion**: it does not touch the encoder or attention blocks of any anchor, so no expensive multi-model forward pass is needed if we cache each anchor’s output once per batch.

2. **New ensemble trainer**  
   - Create `experiments/train_ensemble_anchors_mpiinf3dhp.py`  
     - Loads frozen checkpoints:
       - Temporal anchor: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
       - Factorized anchor: `RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint` (after the full-capacity run)
       - Visibility-v2 anchor: `CrossviewResidualVisibilityV2` (revisited only if training is fixed)
     - Trains **only** the ensemble weights / residual head for 3–5 epochs on MPI-INF-3DHP train.
     - Loss: standard MPJPE on the fused output; optional PA-MPJPE auxiliary loss with weight 0.1.
     - Optimizer: Adam, lr=1e-3, no scheduler.

3. **Exact files to create or modify**
   - **Create** `motionflow_mv/fusion/ensemble_anchor_model.py`
   - **Create** `experiments/train_ensemble_anchors_mpiinf3dhp.py`
   - **Create** `experiments/eval_ensemble_anchors_mpiinf3dhp.py`
   - **Create** `scripts/run_ensemble_anchors_smoke_wsl.sh`
   - **Modify** `motionflow_mv/eval/benchmark_protocol.py` (optional): expose a helper `load_pretrained_anchor(name, checkpoint_path, device)` so the trainer can reuse the existing model-loading logic.

### 3.2 Ensemble Rules

- Each constituent model is kept **frozen** during ensemble training to avoid destabilizing already-converged anchors.
- Weights are initialized so the temporal anchor receives 0.6, factorized 0.3, visibility-v2 0.1 (softmax-normalized).
- If a checkpoint is missing, the ensemble gracefully falls back to averaging the available models.

---

## 4. Smoke-Test Plan

Run a 3–5 epoch smoke on a 500-sample subset of MPI-INF-3DHP train, using only the **temporal anchor** and a **random-initialized dummy factorized branch** (to simulate the factorized anchor before it is ready).

| Step | Command / Script | Pass Criteria | Fail Criteria |
|---|---|---|---|
| 1. Sanity forward | `python experiments/train_ensemble_anchors_mpiinf3dhp.py --smoke --epochs 3 --samples 500` | Script runs without NaNs/crashes; ensemble output shape matches `(B, T, J, 3)` | Any crash, NaN, or shape mismatch |
| 2. Overfit tiny batch | Same script with `--samples 32 --epochs 50` | Training loss decreases monotonically for ≥ 40 epochs | Loss plateaus or increases for 10 consecutive epochs |
| 3. Runtime check | `python experiments/benchmark_runtime.py --model ensemble_anchors` | Ensemble forward latency ≤ 1.1× the slowest single-anchor latency | Latency > 1.5× slowest anchor |
| 4. Integration check | `python experiments/eval_ensemble_anchors_mpiinf3dhp.py --checkpoint tmp/ensemble_smoke.pth` | Produces `outputs/ensemble_smoke_metrics.json` with finite MPJPE | Missing output or infinite/NaN metrics |

**Pass/fail summary:**  
- **Pass:** 3–5 epoch smoke finishes in < 15 min on RTX 4090, metrics are finite, and the ensemble is at least no worse than the best single-anchor baseline on the smoke subset.  
- **Fail:** Crash, NaN, or MPJPE > 1.05× the temporal anchor alone.

---

## 5. Evaluation Plan

1. **Primary metrics** on MPI-INF-3DHP clean val/test:
   - MPJPE, PA-MPJPE, PCK@50/100/150, AUC
   - Report: single-anchor temporal baseline vs. ensemble (average) vs. ensemble (residual).
   - Success: clean MPJPE < 9.0 mm and PA-MPJPE < 5.2 mm (≥ 3% improvement over 9.32 / 5.37).

2. **Robustness matrix**  
   - Re-use `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` with a new `--model ensemble_anchors` path.  
   - Compare ensemble vs. temporal anchor on the 6-axis matrix from iter13; expect the ensemble to be more robust on `rot_*`, `cxcy_*`, and `view_dropout_*` because the factorized branch and visibility-v2 branch provide complementary failure modes.

3. **Latency / throughput**  
   - Use `experiments/benchmark_runtime.py` to report single-frame and single-clip latency, including cache-warm and cache-cold modes.

4. **Scripts**
   - `scripts/run_ensemble_anchors_smoke_wsl.sh` — smoke test
   - `experiments/eval_ensemble_anchors_mpiinf3dhp.py` — full eval
   - `experiments/compare_sota_baselines.py --ensemble` — regenerate SOTA table row

---

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Duration | Notes |
|---|---|---|---|
| Code creation + unit tests | CPU | < 2 h | No GPU needed |
| Smoke train (3–5 epochs, 500 samples) | RTX 4090 | ~10–15 min | One model in, ensemble head only |
| Full ensemble train (5 epochs, full MPI-INF-3DHP) | RTX 4090 | ~2–4 h | Constituent models kept frozen |
| Full eval + robustness matrix | RTX 4090 / CPU | ~30–60 min | Mostly inference; robustness matrix is CPU-light |

**Total active GPU time:** ≤ 5 h once the factorized and visibility-v2 checkpoints are ready.  
**CPU time:** negligible beyond standard data loading.

---

## 7. Risks & Fallback

| Risk | Likelihood | Mitigation / Fallback |
|---|---|---|
| Factorized anchor is still far from 9.32 mm when its full run completes | Medium | Use only the temporal + visibility-v2 pair, or fall back to a temporal-only baseline with no runtime penalty. |
| Ensemble overfits to clean val because all anchors are correlated | Medium | Add dropout to the residual head, freeze the average weights, or constrain the residual magnitude to ≤ 10 mm. |
| Loading 3 large models exceeds RTX 4090 VRAM | Low | Cache predictions offline to `.npy` first, then train the ensemble on CPU. |
| Visibility-v2 training remains CPU-bound / unavailable | High | Exclude visibility-v2 from the initial ensemble; the proposal still yields a temporal + factorized ensemble. |
| Residual head makes metrics worse than simple averaging | Low | Default to the scalar-weighted average variant; the residual head is optional. |
| No reproducible improvement over 9.32 mm | Medium | Treat the experiment as negative evidence; document failure mode and disband ensemble work rather than tuning indefinitely. |

---

## 8. Next Action

Create `motionflow_mv/fusion/ensemble_anchor_model.py` and the smoke trainer, then run the 3-epoch smoke on the temporal anchor + dummy factorized branch as soon as the factorized full-capacity run is queued.
