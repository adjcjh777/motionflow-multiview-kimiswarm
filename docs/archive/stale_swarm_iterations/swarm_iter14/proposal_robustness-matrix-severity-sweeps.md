# Proposal: Robustness Matrix Severity Sweeps

**Author:** iter14 swarm agent (robustness/evaluation)  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — 9.32 mm MPJPE / 5.37 mm PA-MPJPE on MPI-INF-3DHP S2/Seq1 clean  
**Related prior work:** `docs/swarm_iter13_next_iteration_synthesis.md` §3.4, `docs/results_iter13.md`

---

## 1. Problem

The existing 6-axis robustness matrix (`experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`) samples each corruption axis at only one or two fixed severities, so it cannot reveal at which severity the model transitions from acceptable to catastrophic, nor can it test combined corruptions or compare the new visibility-v2/factorized checkpoints on identical conditions.

## 2. Hypothesis

If we sweep severity levels for each of the six corruption axes and add a small combined-corruption set, we will obtain a denser robustness surface that identifies the exact failure thresholds of the PP baseline, visibility-v2, and factorized-PP models, and shows whether visibility gating improves graceful degradation under occlusion.

## 3. Method

### 3.1 New evaluation script

Create `experiments/eval_robustness_matrix_severity_mpiinf3dhp.py`.

- Reuses the per-condition evaluator from `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` but generalises it to:
  - accept a `--model_class` argument (`pp_baseline`, `visibility_v2`, `factorized_pp`) so one script can compare checkpoints;
  - accept a YAML/JSON severity config (`--severity_config`) defining per-axis severity grids;
  - generate **single-axis sweeps** for rotation, translation, focal length, principal point, radial distortion, and occlusion (view/joint dropout);
  - generate a **combined-corruption set** at low/medium/high mixes (e.g., focal 1% + cxcy 2 px + view dropout 0.2);
  - write a structured JSON manifest and Markdown table per model.

Key files to create / modify:

- **Create** `experiments/eval_robustness_matrix_severity_mpiinf3dhp.py`
  - Imports `motionflow_mv.calibration.perturb.perturb_intrinsics`, `perturb_extrinsics`, `perturb_radial_distortion`.
  - Imports occlusion helpers from `motionflow_mv.data.occlusion_aug`.
  - Uses `motionflow_mv.eval.benchmark_protocol.BenchmarkProtocol` for metric computation.
  - Supports `--num_clips 20` smoke mode and `--num_clips -1` full mode.
- **Create** `configs/robustness_severity_sweep.yaml` — default severity grids for the six axes.
  ```yaml
  axes:
    rotation: [0.0, 0.25, 0.5, 1.0, 2.0]  # deg std
    translation: [0.0, 0.005, 0.010, 0.020]  # m std
    focal: [0.0, 0.01, 0.02, 0.05]          # relative
    pp: [0.0, 1.0, 2.0, 3.0, 5.0]           # px
    distortion_k1: [0.0, 0.01, 0.05, 0.10]
    view_dropout: [0.0, 0.1, 0.2, 0.3, 0.4]
    joint_dropout: [0.0, 0.1, 0.2, 0.3, 0.4]
  combined:
    - {rotation: 0.5, focal: 0.01, view_dropout: 0.2}
    - {translation: 0.010, pp: 2.0, joint_dropout: 0.2}
    - {rotation: 1.0, focal: 0.02, pp: 3.0, view_dropout: 0.3}
  ```
- **Create** `scripts/run_robustness_severity_sweep_wsl.sh` — smoke/full launcher for WSL/RTX 4090.
- **Modify** `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` (minor)
  - Extract reusable `evaluate_condition(model, loader, cfg, device)` into a shared helper module, or at minimum add a `--num_clips` flag and a `--model_class` flag so the old script remains the canonical single-model path.
- **Modify** `experiments/plot_robustness_matrix.py` (minor)
  - Add `--mode severity` to render per-axis line plots and a combined heatmap from the new JSON schema.

### 3.2 Severity sweep protocol

1. **Clean baseline:** run the script with all severities set to 0; verify anchor MPJPE 9.32 mm.
2. **Single-axis sweeps:** for each axis, sweep from 0 to the maximum in the config.
3. **Combined corruptions:** run the three predefined mixes.
4. **Cross-model comparison:** run identical configs on the PP baseline, visibility-v2, and factorized-PP checkpoints.
5. **Outputs:**
   - `outputs/robustness_matrix_severity_{model}.json`
   - `docs/tables/icra2027/robustness_matrix_severity_{model}.md`
   - `docs/figures/robustness_matrix_severity_{model}.png`

### 3.3 Integration with existing infrastructure

- Uses the existing `TemporalClipDataset` and `collate_fn` from `eval_robustness_matrix_pp_mpiinf3dhp.py`.
- Uses `motionflow_mv.eval.metrics.compute_all_metrics` for MPJPE, PA-MPJPE, PCK, AUC.
- Does not touch training code, loss functions, or model definitions.

## 4. Smoke-Test Plan

Because this is a pure evaluation proposal, the "3–5 epoch" smoke is implemented as **3 repeated passes (pseudo-epochs) over a 20-clip subset** to verify deterministic output and no leakage/crashes.

- **Dataset:** MPI-INF-3DHP S2/Seq1, first 20 clips only (≈50 batches with `batch_size=8`).
- **Sweeps:** 3 severity levels per axis plus 2 combined conditions (low/medium).
- **Pass criteria:**
  1. Clean MPJPE is within 0.5 mm of the anchor (9.32 mm) on the subset.
  2. No NaNs, no OOM, and deterministic outputs across the 3 passes (same JSON to within floating-point tolerance).
  3. Single-axis sweep completes in ≤ 10 minutes on RTX 4090 / CPU.
  4. The JSON and Markdown artifacts are produced with keys `mpjpe`, `pa_mpjpe`, `pck@50mm`, `pck@100mm`, `pck@150mm`, `pck_auc` for every condition.
- **Fail criteria:**
  - Clean MPJPE deviates > 0.5 mm.
  - Any axis crashes or produces non-finite metrics.
  - Runtime > 15 minutes on the subset.

## 5. Evaluation Plan

### Metrics

For every severity level and combined condition, report:
- MPJPE (mm)
- PA-MPJPE (mm)
- PCK@50mm / @100mm / @150mm
- PCK-AUC (0–150 mm)
- Per-joint MPJPE/PA-MPJPE (stored in JSON for diagnostic plotting)

### Scripts

1. Smoke:
   ```bash
   bash scripts/run_robustness_severity_sweep_wsl.sh smoke
   ```
   Equivalent to:
   ```bash
   python experiments/eval_robustness_matrix_severity_mpiinf3dhp.py \
       --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
       --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
       --model_class pp_baseline \
       --num_clips 20 \
       --config configs/robustness_severity_sweep.yaml \
       --out_json outputs/robustness_matrix_severity_pp_smoke.json \
       --out_md docs/tables/icra2027/robustness_matrix_severity_pp_smoke.md
   ```
2. Plot:
   ```bash
   python experiments/plot_robustness_matrix.py \
       --input outputs/robustness_matrix_severity_pp_smoke.json \
       --output docs/figures/robustness_matrix_severity_pp_smoke.png \
       --mode severity
   ```
3. Full sweep:
   ```bash
   bash scripts/run_robustness_severity_sweep_wsl.sh full
   ```

### Success criteria for full evaluation

- Clean MPJPE ≤ 9.6 mm on the full S2/Seq1 split.
- No moderate-severity single-axis condition degrades > 30% relative to clean.
- Combined low-severity MPJPE ≤ 15 mm.
- All visibility-v2/factorized-PP checkpoints run under identical conditions for direct comparison.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Mode | Data | Conditions | Estimated Runtime | GPU/CPU |
|------|------|------------|-------------------|---------|
| Smoke | 20 clips, 3 pseudo-epochs | 6 axes × 3 levels + 2 combined ≈ 38 conditions | ~5–10 min | CPU OK, GPU optional |
| Full single-model | S2/Seq1 full (~2k clips) | 6 axes × 5 levels + 3 combined ≈ 53 conditions | ~30–60 min | RTX 4090 / CPU |
| Full 3-model comparison | Same as above ×3 | Same | ~1.5–3 h | RTX 4090 preferred for learned-model inference |

Memory: < 8 GB VRAM at `batch_size=8`, `clip_len=13`. CPU fallback is feasible but ~3× slower for learned-model inference.

## 7. Risks & Fallback

| Risk | Impact | Fallback |
|------|--------|----------|
| Combined corruptions produce degenerate camera matrices (e.g., cxcy 5 px + rotation 2°) causing numerical instability or NaN triangulation | Metrics become invalid | Clip per-axis severity in combined sets; skip any condition with non-finite metrics and report as "degenerate" |
| Visibility-v2 or factorized-PP checkpoints have incompatible model signatures | Script crashes for some `--model_class` values | Default to PP baseline only; add adapter classes that wrap each signature to a common `(x, K, R, t)` interface |
| Severity sweep multiplies runtime beyond paper deadline | Cannot finish 3-model comparison | Run single-axis sweeps only for the two highest-priority axes (occlusion, focal/PP) and defer full sweep to post-deadline |
| Subset smoke is too small and gives noisy pass/fail | False positive / false negative | Use 3 seeds for dropout axes and report mean/std; keep clean-subset threshold at 0.5 mm |
| Plotting script cannot consume the new JSON schema | No figures for paper | Write a tiny conversion script `experiments/convert_robustness_json.py` to flatten severity results into the old schema |

---

## Summary

This proposal extends the existing single-point robustness matrix into a **severity sweep + combined-corruption benchmark** that can evaluate any checkpoint (PP baseline, visibility-v2, factorized-PP) on identical conditions. It is eval-only, requires no new model training, and is designed to finish within one hour on an RTX 4090 for the full sweep.
