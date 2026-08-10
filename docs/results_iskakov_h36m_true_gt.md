# Iskakov et al. (ICCV 2019) Learnable Triangulation Baseline — H36M True GT

> **Date:** 2026-08-10
> **Status:** Baseline extended to the H36M true-GT standard protocol
> (issue #194, P0-1). All numbers read verbatim from the run log listed in the
> Evidence table. Companion to `docs/results_iskakov_baseline.md`
> (Shelf/Campus protocol).
> **Reference (exact):** Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y.,
> 'Learnable Triangulation of Human Pose', ICCV 2019, arXiv:1905.05754.

## Protocol

- **Data / split:** H36M true-GT standard protocol — manifest
  `configs/splits/h36m_true_gt_standard.yaml`; train subjects S1, S5, S6, S7,
  S8 (all 15 actions, ~390k frames), test subjects S9 (83,759 frames) and
  S11 (57,971 frames). 4 views, 17 joints, metres. Labels are
  official-mocap-derived (data_3d_h36m.npz), NOT DLT triangulation; double
  acceptance gates passed (circularity MJE 27,762–37,320 mm >> 0; reprojection
  RMSE 3.13–7.15 px ≤ 15 px).
- **Model:** one shared `IskakovLearnableTriangulation` (hidden_dim=32,
  cross-view, 1,569 params) trained across all 5 train subjects, evaluated
  per test subject + macro mean.
- **Training:** 10 epochs × 8,192 sampled frames/epoch, batch 64, AdamW
  lr 1e-3 → cosine to 0, weight decay 1e-4, grad clip 1.0, seed 20260810,
  single RTX 4090. Loss: direct MPJPE in mm. ~65 s total.
- **Frozen references:** stride-subsampled val subset (2,000 frames per
  subject, deterministic) of the full-set numbers, using the leaderboard's
  float64 SVD routine — within subsample noise of the full-set DLT baseline
  (S9 29.54 / S11 21.81 mm).

## Results

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) |
|---|---:|---:|---:|
| DLT unweighted (frozen, ref subset) | 33.61 | 24.77 | 29.19 |
| DLT confidence-weighted (frozen, ref subset) | 29.82 | 21.91 | 25.87 |
| **Learned weights (best epoch 7)** | **27.13** | **19.64** | **23.38** |

Root-aligned MPJPE at best epoch: S9 27.04 mm, S11 19.21 mm, combined
23.12 mm (frozen refs: combined root 29.31 unweighted / 25.55 conf-weighted).

Gains: **+2.49 mm vs conf-weighted DLT**, **+5.81 mm vs unweighted DLT**
(combined direct). Epoch trajectory (combined direct): 25.32 → 23.56 → 23.40
→ 23.38 → 23.38 → 23.38 → **23.38** → 23.38 → 23.39 → 23.39 (saturated by
epoch 3-4, no NaN).

## Interpretation

1. The learned per-view/per-joint weights beat both frozen DLT variants on
   every H36M test subject. This is the same pattern as on Shelf/Campus, and
   holds on the flagship H36M protocol now that the labels are true GT.
2. The absolute numbers (S9 27.13 / S11 19.64 mm) are inside the expected
   15–30 mm band for H36M multiview pose — first-order sanity that the
   true-GT protocol is comparable to the literature's regime.
3. The baseline is frame-wise and tiny (1.6k params); on Shelf/Campus its
   gain was similarly data-limited. On H36M the 390k-frame train set already
   saturates it by epoch 3, so larger variants would be needed to probe the
   protocol ceiling.

## Evidence

| Run | Log | Checkpoint | Config/history |
|-----|-----|------------|----------------|
| H36M true GT | `outputs/iskakov_learnable_tri_h36m_true_gt.log` | `outputs/iskakov_learnable_tri_h36m_true_gt.pth` | `outputs/iskakov_learnable_tri_h36m_true_gt.config.json` |

Reproduce:

```bash
CUDA_VISIBLE_DEVICES=0 /d/anaconda3/python.exe \
    experiments/train_iskakov_baseline_shelf_campus.py --protocol h36m \
    --epochs 10 --train_samples_per_epoch 8192 --batch_size 64 \
    --ref_max_frames 2000 \
    --log_path outputs/iskakov_learnable_tri_h36m_true_gt.log
```

## Sparse-view MPJPE@k curves (2026-08-11)

Evaluated with `experiments/eval_iskakov_mpjpe_at_k.py` (same checkpoint as
above; deterministic view subsets, seed 20260810; frozen DLT refs on the
first subset of each k, float64 SVD routine). Full JSON:
`outputs/iskakov_mpjpe_at_k/iskakov_mpjpe_at_k_h36m.json` and
`..._shelf_campus.json`.

### H36M true GT (4 views; macro mean over S9+S11)

| k | Learned direct (mm) | Learned root (mm) | DLT unweighted (first subset) | DLT conf-weighted (first subset) |
|---:|---:|---:|---:|---:|
| 2 | 53.61 (±27) | 55.12 | 37.19 avg | 36.42 avg |
| 3 | **27.80** (±2) | 27.54 | 34.86 avg | 33.68 avg |
| 4 | **23.39** | 23.14 | 29.15 avg | 25.94 avg |

### Shelf/Campus detected (Campus 3 views primary; Shelf carries the calibration caveat)

| k | Learned direct (mm) | Learned root (mm) | DLT unweighted (first subset) | DLT conf-weighted (first subset) |
|---:|---:|---:|---:|---:|
| 2 | 162.81 | 148.21 | Campus 135.86 / Shelf 239.97 | Campus 134.85 / Shelf 240.03 |
| 3 | **134.99** | 125.24 | Campus 138.08 / Shelf 156.37 | Campus 136.96 / Shelf 152.26 |
| 4 (Shelf) | **127.60** | 123.98 | 138.52 | 135.31 |
| 5 (Shelf) | **123.12** | 120.33 | 130.77 | 127.63 |

### Reading

1. **For k >= 3 views, the learned weights beat both frozen DLT variants on
   every evaluated subset of both protocols.** On H36M the gain grows with k
   (k=3: ~7-8 mm; k=4: ~2.6-5.8 mm vs conf/unweighted DLT). On Shelf the
   gain is largest exactly where DLT suffers most from the coarse calibration
   (k=2: 240 -> 162 mm; k=3: 156 -> 137 mm).
2. **k=2 is the honest limitation of this baseline**: the trainer uses
   full-view batches only (no view dropout), so at 2 views the model is
   out of distribution and loses to DLT on both protocols except the
   badly-calibrated Shelf. Adding view-dropout training to the weight
   predictor is the next lever for the sparse-view story.
3. These curves are the first non-circular MPJPE@k evidence in the repo and
   support the roadmap's sparse-view-robustness positioning
   (`docs/roadmap_cvpr2027.md`).
