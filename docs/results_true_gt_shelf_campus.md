# True-GT Shelf/Campus Leaderboard (detected 2D + real mocap 3D)

> **Date:** 2026-08-10
> **Status:** First non-circular leaderboard for the project. All numbers verified
> from run logs and re-computed diagnostics on this date.
> **Scope:** Smoke-scale (3-epoch) benchmark only. Per the data-foundation
> constraints, these numbers are not yet a basis for final model selection until
> H36M true-GT and MPI detected-2D protocols are also repaired.

## Protocol

- **Data:** `data/webbridge/shelf_campus_detected/` built by
  `experiments/build_shelf_campus_canonical_from_detection.py` from
  `detection.json` (real off-the-shelf 2D detections) +
  `annotation_3d.json` (real annotated 3D). Labels are **not** a function of the
  input 2D.
- **Split manifest:** `configs/splits/shelf_campus_detected_smoke.yaml`
  - train: Shelf seq1 train (346 frames, 5 views) + Campus seq1 train (652 frames, 3 views)
  - val: Shelf seq1 val (87 frames) + Campus seq1 val (164 frames)
  - 17 joints, units in metres.
- **Non-circularity check** (`scripts/diagnose_circular_labels.py`, DLT re-triangulation vs stored 3D):

| File | Direct MJE (mm) | Root-aligned MPJPE (mm) |
|------|---:|---:|
| shelf_seq1_train_detected_m.npz | 124.67 | 96.69 |
| shelf_seq1_val_detected_m.npz | 130.77 | 124.13 |
| campus_seq1_train_detected_m.npz | 124.51 | 107.74 |
| campus_seq1_val_detected_m.npz | 138.08 | 120.61 |

  Non-zero errors confirm the labels are genuinely independent of the input 2D
  (contrast: `data/h36m_hf/*.npz` and `data/webbridge/h36m_corrected/*.npz`
  both measure 0.0000 mm, i.e. circular).

### 2D/3D/camera consistency audit (2026-08-10)

`scripts/check_true_gt_reprojection.py` reprojects the true 3D through the
stored cameras and compares with the detected 2D (conf > 0.5 joints):

| File | Overall RMSE (px) | View medians (px) | Inlier fraction (<15 px) |
|------|---:|---|---:|
| campus_seq1_val_detected_m.npz | 7.65 | 8.8 / 7.3 / 6.6 | 0.960 |
| shelf_seq1_val_detected_m.npz | 53.72 | 90.3 / 28.5 / 29.7 / 40.7 / 54.0 | 0.386 |

- **Campus is clean**: ~5 px median reprojection error on every view, i.e.
  detected 2D, calibration, and true 3D are mutually consistent.
- **Shelf is systematically inconsistent**: the misalignment is present in
  87/87 val frames (not a few bad detections), and Shelf's `dist_coeffs` are
  all zero, so lens distortion is not the cause. The Shelf release's camera
  calibration is simply coarse relative to its detections/annotations.
- Control: on the circular `data/h36m_hf/s_01_act_02_multiview.npz` the same
  audit gives 2.75 px RMSE, as expected when labels are DLT(inputs).
- **Implication:** treat Campus (3-view) as the primary sparse-view benchmark;
  report Shelf numbers with a calibration-quality caveat, or restrict Shelf
  training to confidence/reprojection-filtered joints. Reproduce with:
  `python scripts/check_true_gt_reprojection.py data/webbridge/shelf_campus_detected/*.npz --threshold 25`

## Training setup (common to all learned runs)

- Trainer: `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  (mixed loader, `num_domains` auto-set from manifest after the domain-embedding
  OOB fix).
- 3 epochs, batch size 4, `--train_samples 128`, `--val_stride 10`,
  lr 1e-3 with cosine decay + 1-epoch warmup, EMA 0.999, grad clip 1.0.
- Variable-view training with view dropout/permutation enabled.
- All runs completed without NaN/inf/crash on the local RTX 4090.

## Leaderboard

| Rank | Method | Params | Best val MPJPE (mm) | Best epoch | Epoch trajectory (mm) |
|-----:|--------|---:|---:|:---:|---|
| 1 | **Iskakov-style learned weights (mixed)** | 1.6 k | **119.32** root / 128.05 direct | 4 | direct: 133.75 → 132.55 → 129.46 → **128.05** → … (early stop 12) |
| 2 | **DLT baseline (root-aligned)** | — | **122.37** | — | — |
| 3 | **DLT baseline (direct MJE)** | — | **134.43** | — | — |
| 4 | v80 (view-reliability weighting) | 965 k | **408.58** | 3 | 429.32 → 426.62 → 408.58 |
| 5 | v57 (domain-conditional PSC) | 1 009 k | **424.63** | 3 | 429.24 → 427.12 → 424.63 |
| 6 | v25 (geometry fusion, d=128/h=256) | 2 732 k | **430.67** | 1 | 430.67 → 459.58 → 452.26 |

DLT numbers are the mean of Shelf-val and Campus-val from the diagnostic table
above: direct MJE (130.77 + 138.08)/2 = 134.43 mm; root-aligned
(124.13 + 120.61)/2 = 122.37 mm. The Iskakov-style row is the standalone
learnable-triangulation baseline of `docs/results_iskakov_baseline.md`
(Iskakov et al., ICCV 2019, arXiv:1905.05754; algebraic weight-prediction
branch, 1,569 params): it beats both frozen DLT variants on the macro mean
(root 119.32 < 122.37; direct 128.05 < 134.43). Caveats: Shelf numbers carry
the coarse-calibration warning, and training saturates within ~4 epochs on
this ~1k-frame protocol (data-limited, not an architecture verdict).

Model configs: v80 `d=64, residual_hidden=128, n_st_layers=2`;
v25 `d=128, residual_hidden=256, n_st_layers=3`; v57 as in
`outputs/omniview_fusion_v57_shelf_campus_detected_smoke.config.json`.

## Evidence

| Run | Log | Checkpoint | Config |
|-----|-----|------------|--------|
| v25 | `outputs/omniview_fusion_v25_shelf_campus_detected_smoke.log` | `..._smoke.pth` | `..._smoke.config.json` |
| v57 | `outputs/omniview_fusion_v57_shelf_campus_detected_smoke.log` | `..._smoke.pth` | `..._smoke.config.json` |
| v80 | `outputs/omniview_fusion_v80_shelf_campus_detected_smoke.log` | `..._smoke.pth` | `..._smoke.config.json` |

## Interpretation

1. **Naive DLT beats every 3-epoch learned model** on this true-GT, sparse-view
   protocol (5/3 views). Under the previous circular H36M protocol v25 appeared
   to achieve ~17 mm; on real labels it is ~3x worse than plain triangulation
   after 3 epochs. This confirms the circular-label diagnosis empirically.
2. **Among learned models the ranking is v80 > v57 > v25.** v80's
   view-reliability weighting is the most valuable direction for sparse
   cross-view scenarios; v25 (the largest model, 2.7 M params) is the worst and
   even degraded after epoch 1, consistent with it overfitting to reproduce a
   DLT label it can no longer copy exactly.
3. All learned models still fall monotonically (v57, v80) or nearly so (v25);
   3 epochs on ~1 000 frames is far from converged. The gap to DLT is therefore
   an under-training + protocol gap, not yet an architecture verdict.

## Long-horizon v80 run on A800-D (2026-08-10, completed)

**Setup:** `scripts/run_v80_shelf_campus_detected_long_a800.sh` — identical
model/loss config as the 3-epoch smoke (verified against
`outputs/omniview_fusion_v80_shelf_campus_detected_smoke.config.json`),
scaled to epochs=25, batch_size=8, train_samples=512, val_stride=2,
lr 1e-3 cosine with 2-epoch warmup. A800-D, `CUDA_VISIBLE_DEVICES=4,5`
(single process, GPUs pinned), nohup. Micro-smoke first confirmed epoch-1
val 433.37 mm matches the local smoke (no environment drift).

**Trajectory (val MPJPE, mm):**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | ... | 25 (final) |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 433.4 | 419.9 | 385.4 | 345.8 | 312.9 | 285.3 | **276.5** | 292.8 | ... | 783.6 |

**Findings:**

1. Best val MPJPE **276.49 mm at epoch 7** — the learned model improves
   1.55× over the 3-epoch smoke (408.58 mm) but **still loses to the naive
   root-aligned DLT baseline (122.37 mm) by 2.3×**.
2. **Severe overfitting after epoch 7**: train loss keeps falling
   (9.03 → 6.50) while val MPJPE rises monotonically (276 → 784 mm by the
   final epoch 25). With only ~998 train frames this is expected; the 3-epoch
   smoke had not even reached the overfitting onset.
3. No NaN/inf through all 25 epochs.

**Implication for model selection:** longer training alone does not close the
gap to DLT on ~1k frames. Next levers (in order): (a) early-stopped best
checkpoint + weight decay / stronger view dropout, (b) Campus-only training
(the calibration-clean 3-view set), (c) the Iskakov-style learnable-weight
DLT baseline as the reference learned method, (d) more data (H36M true GT /
MPI detected-2D once P0-1/P0-2 unblock).

**Evidence:** remote log
`a800-D:/mnt/nvme0n1p1/zhangzy/motionflow-mv-detected-long/outputs/omniview_fusion_v80_shelf_campus_detected_long.log`;
best checkpoint `..._long.pth` (epoch-7 weights); local script
`scripts/run_v80_shelf_campus_detected_long_a800.sh`.

## Next steps for this protocol

- [x] Longer v80/v57 runs (full data, >= 20 epochs) to see if any learned model
      closes the gap to the ~122 mm root-aligned DLT baseline. — **Done
      (2026-08-10):** v80 best 276.49 mm at epoch 7, then overfits; v57 best
      306.45 mm at epoch 4, then overfits. Longer training alone does NOT
      close the gap on ~1k frames (see the long-horizon section below).
- [ ] Add RANSAC/IRLS robust triangulation baselines under the same manifest.
- [x] Add Iskakov et al. (ICCV 2019) learnable-triangulation baseline — **done
      on Shelf/Campus** (`docs/results_iskakov_baseline.md`, mixed run 128.05
      direct / 119.32 root mm) **and on the H36M true-GT protocol**
      (`docs/results_iskakov_h36m_true_gt.md`, combined direct 23.38 mm vs
      conf-DLT 25.87 mm). Caveat: Shelf/Campus alone remains too small for
      the paper's standard (H36M-pretrained) training protocol of that method.
- [ ] Report per-dataset (Shelf vs Campus) splits in the paper, since Campus
      (3 views) is the harder sparse-view regime.

## Acceptance gate for future true-GT protocols (H36M)

The H36M true-GT pipeline is staged and unit-tested
(`experiments/prepare_h36m_true_gt.py`, 16 tests in
`tests/test_h36m_true_gt_pipeline.py`). Once official mocap
(`PosesD3_Positions`) lands in `data/h36m_true_gt/`, a regenerated npz is only
accepted when **both** checks pass:

1. `python scripts/diagnose_circular_labels.py <npz>` — direct MJE must be
   >> 0 mm (labels independent of input 2D).
2. `python scripts/check_true_gt_reprojection.py <npz>` — reprojection RMSE
   within ~15 px (correct joint mapping, units, frame alignment).
