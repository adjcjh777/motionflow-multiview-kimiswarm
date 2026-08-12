# Multi-Scale Temporal Fusion: Fuse Clip Windows at Multiple Temporal Scales

**Date:** 2026-08-06  
**Author:** MotionFlow-MultiView planning swarm (agent: multi-scale temporal fusion)  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Status:** Proposal, not yet implemented

---

## 1. Problem

The current anchor processes a clip at a single fixed temporal scale (e.g. 9 or 13 frames), so it misses either fine-grained fast motion (short windows) or long-range temporal context (long windows).

## 2. Hypothesis

Fusing per-frame pose estimates from multiple overlapping temporal windows (short / medium / long) with a learned scale-mixing head will improve robustness to fast motion and occlusions while keeping clean accuracy within 0.5 mm of the anchor.

## 3. Method

### 3.1 Architecture changes

Create a new model file:

- **`motionflow_mv/fusion/multiscale_temporal_clip_fusion_model.py`**
  - Wraps the existing `RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint` (the factorized PP baseline) as a *temporal backbone*.
  - Accepts a long input clip, e.g. `T = 27` frames, and extracts three overlapping window banks:
    - `T_short = 5` frames, stride 2
    - `T_medium = 13` frames, stride 4
    - `T_long = 27` frames, stride 8 (whole clip)
  - Each window bank is fed independently through the shared backbone, producing per-window 3-D pose predictions and confidences.
  - A lightweight **temporal scale fusion head**:
    - Projects predictions from each scale to a common `(T, J, 3)` timeline using linear interpolation at frame centers.
    - Stacks scale-indexed embeddings + per-frame confidences.
    - Applies a 1-D temporal convolution (kernel 3) + layer norm to blend scales.
    - Outputs final per-frame 3-D pose.
  - Keeps the existing PP/focal correction head unchanged; corrected intrinsics are computed once on the long clip and reused for all sub-windows.

### 3.2 Training / loss changes

Create a new training script:

- **`experiments/train_multiscale_temporal_clip_fusion_mpiinf3dhp.py`**
  - Reuses `RandomClipDataset` with `clip_len=27`.
  - Loss is standard per-frame MPJPE (L2) on the fused output, averaged over the central 13 frames to avoid boundary effects.
  - Optional auxiliary loss (weight 0.1): per-scale MPJPE, encouraging each scale to be individually reasonable.

### 3.3 Data changes

- No new dataset required.
- Use existing MPI-INF-3DHP `.npz` files under `data/webbridge/mpi_inf_3dhp/`.
- The model expects clips of length `27`; shorter clips are padded / center-cropped by the dataset loader.

### 3.4 Exact files to create or modify

**Create:**

1. `motionflow_mv/fusion/multiscale_temporal_clip_fusion_model.py`
2. `experiments/train_multiscale_temporal_clip_fusion_mpiinf3dhp.py`
3. `experiments/eval_multiscale_temporal_clip_fusion_mpiinf3dhp.py`
4. `scripts/run_multiscale_temporal_clip_fusion_smoke_wsl.sh`

**Modify (proposed exact edits):**

- `motionflow_mv/fusion/__init__.py`: add `MultiScaleTemporalClipFusionModel` to the public model registry.
- `experiments/benchmark_runtime.py`: register the new model under a short alias `mstcf` so the latency comparison against the anchor is apples-to-apples.
- `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`: add a `--model mstcf` path (optional, only after smoke passes).

## 4. Smoke-Test Plan

Run a 3–5 epoch smoke on a small sample (500 clips, `T=27`).

```bash
bash scripts/run_multiscale_temporal_clip_fusion_smoke_wsl.sh
```

**Pass criteria:**

- Clean val MPJPE ≤ 10.0 mm (anchor is 9.32 mm; allow ≤ 0.7 mm regression for a tiny smoke run).
- No NaNs, no `RuntimeError`, and training completes in < 30 min on RTX 4090.
- Multi-scale fusion head reduces single-scale MPJPE by at least 5% on the val set.

**Fail criteria:**

- Clean val MPJPE > 11.0 mm or any NaN/instability.
- Multi-scale fusion does not outperform the average of the three single-scale baselines.
- Memory usage exceeds 12 GB on a batch size of 2.

## 5. Evaluation Plan

After the smoke, run full evaluation:

1. **Clean metrics** on MPI-INF-3DHP val/test:
   - `experiments/eval_multiscale_temporal_clip_fusion_mpiinf3dhp.py`
   - Metrics: MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
   - Target: MPJPE ≤ 9.6 mm, PA-MPJPE ≤ 5.7 mm (within ~3% of anchor).

2. **Robustness matrix** (6-axis corruption):
   - Reuse `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` with `--model mstcf`.
   - Compare against the anchor on: view dropout, joint dropout, focal perturbation, PP perturbation, rotation, translation.

3. **Latency comparison**:
   - `python experiments/benchmark_runtime.py --model mstcf --clip_len 27`
   - Target: throughput within 50% of the anchor at the same clip length; if slower, document the accuracy-vs-latency trade-off.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time | Memory |
|---|---|---|---|
| 5-epoch smoke (500 clips) | RTX 4090 | ~25–30 min | ~8–10 GB |
| Full 40-epoch training | RTX 4090 | ~18–24 h | ~10–12 GB |
| Clean eval | CPU or GPU | < 5 min | < 2 GB |
| Robustness matrix | CPU/GPU eval | ~15 min | < 2 GB |

The multi-scale approach is ~1.3–1.5× slower per clip than the single-scale anchor because it processes three window banks; the fusion head itself is negligible.

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|---|---|
| **Alignment complexity**: fusing three temporal scales may mis-align frames near clip boundaries. | Use center-frame interpolation and evaluate only the central 13 frames for smoke; if it fails, fall back to a two-scale (short+long) design. |
| **Overfitting on small smoke sample**: 500 clips may not stabilize the fusion head. | Increase to 2k clips or pre-train the backbone frozen for the first 2 epochs. |
| **Memory blow-up**: three forward passes through the backbone. | Process scales sequentially and accumulate features, or reduce `d` to 32 for the smoke. |
| **No accuracy gain**: multi-scale fusion does not beat single-scale. | Keep only the best single scale (likely medium) and treat the experiment as a negative result; no other code is modified. |
| **PP correction saturation**: the learned correction head may saturate under longer clips. | Reuse the intrinsics-curriculum schedule already being tested in the PP robust re-train. |
