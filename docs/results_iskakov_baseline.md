# Iskakov et al. (ICCV 2019) Learnable Triangulation Baseline — True-GT Shelf/Campus

> **Date:** 2026-08-10
> **Status:** First standard SOTA baseline (P0-3) on the non-circular
> Shelf/Campus protocol. All numbers below are read verbatim from the run logs
> listed in the Evidence table. Companion to `docs/results_true_gt_shelf_campus.md`
> (that file is not modified).
> **Reference (exact):** Iskakov, D., Kasneci, E., 'Learnable Triangulation of
> Human Pose', ICCV 2019.

## Protocol

- **Data / split:** identical to the true-GT leaderboard —
  manifest `configs/splits/shelf_campus_detected_smoke.yaml`;
  Shelf seq1 (train 346 / val 87 frames, 5 views) + Campus seq1
  (train 652 / val 164 frames, 3 views), 17 joints, metres.
- **Model:** `motionflow_mv/fusion/iskakov_learnable_triangulation.py` —
  standalone re-implementation of the paper's weight-prediction branch:
  a shared MLP over per-view per-joint features
  `[u_norm, v_norm, confidence, cross-view mean u/v/conf, ray-distance]`
  produces per-view weights in (0,1) via sigmoid; weighted DLT via the
  existing batched lstsq routine
  (`motionflow_mv/fusion/triangulation.py::triangulate_dlt_batched_lstsq`,
  float32 on GPU; matches the leaderboard's float64 SVD routine within
  ≤0.23 mm). Cross-view feature mode (concat each view's features with the
  cross-view mean) is the main variant, matching the paper's joint
  confidence-volume design. MLP final layer zero-initialised, so epoch 0 ==
  unweighted DLT. Params: 1569 (cross-view), 1345 (per-view ablation).
  Documented deviation: the paper feeds deep backbone features; this protocol
  only has raw detections, so features are detection statistics.
- **Training:** `experiments/train_iskakov_baseline_shelf_campus.py`;
  loss = direct MPJPE in mm on metres; AdamW lr 1e-3 (wd 1e-4) with cosine
  decay over all steps; batch 8; grad clip 1.0; domain-balanced sampling for
  the mixed run (each step samples Shelf or Campus with prob 1/2, keeping
  view count constant within a batch); early-stop patience 8 on combined
  direct val MPJPE; seed 20260810; single RTX 4090 (GPU 0). Root-aligned
  (centroid-aligned, identical definition to
  `scripts/diagnose_circular_labels.py`) is logged as a secondary metric.
- **Frozen references (no learning), computed in each log** with the
  leaderboard's SVD routine: unweighted DLT and confidence-weighted DLT on
  val. These reproduce the leaderboard numbers exactly.

## Frozen DLT references (recomputed in this run, identical to leaderboard)

| Val set | Unweighted DLT direct (mm) | Unweighted root (mm) | Conf-weighted direct (mm) | Conf-weighted root (mm) |
|---|---:|---:|---:|---:|
| Shelf ⚠ | 130.77 | 124.13 | 127.63 | 122.03 |
| Campus | 138.08 | 120.61 | 136.96 | 119.87 |
| Macro mean (leaderboard) | **134.43** | **122.37** | 132.29 | 120.95 |

Matches `docs/results_true_gt_shelf_campus.md` (134.43 / 122.37 mm) exactly.

## Results

| Run | Best epoch | Val Shelf direct / root (mm) ⚠ | Val Campus direct / root (mm) | Combined direct / root (mm) | Gain vs conf-DLT (direct) | Gain vs unweighted DLT (direct) |
|---|:---:|---|---|---|---:|---:|
| **Mixed Shelf+Campus, cross-view (main)** | 4 | 123.12 / 120.33 | 132.97 / 118.32 | **128.05 / 119.32** | +4.24 | +6.38 |
| Campus-only, cross-view | 3 | — | **132.34 / 118.17** | 132.34 / 118.17 | +4.61 | +5.74 |
| Campus-only, per-view (ablation, no cross-view features) | 3 | — | 132.63 / 118.45 | 132.63 / 118.45 | +4.33 | +5.46 |

⚠ Shelf caveat: Shelf's release calibration is coarse (reproj RMSE 53.7 px,
see `docs/results_true_gt_shelf_campus.md` §2D/3D audit); Shelf numbers are
reported for completeness only. Campus is the primary benchmark.

Epoch trajectory (mixed run, combined direct MPJPE, mm):
133.75 → 132.55 → 129.46 → **128.05** → 129.01 → 130.18 → 130.95 → 130.34 →
130.91 → 131.65 → 132.49 → 132.30 → early stop at epoch 12.

Campus-only trajectory (direct, mm): 137.54 → 133.70 → **132.34** → 133.10 →
133.22 → 133.26 → 133.24 → 133.27 → 133.81 → 134.12 → 134.57 → early stop at
epoch 11.

All runs completed without NaN/inf/crash on GPU 0.

## Interpretation

1. **Learned weights beat both frozen DLT baselines on every val set.**
   Campus-only direct MPJPE improves from 138.08 mm (unweighted DLT) /
   136.96 mm (conf-DLT) to 132.34 mm (+5.7 / +4.6 mm). The gain is larger on
   Shelf (+7.65 mm direct over unweighted DLT), consistent with the learned
   weights down-weighting Shelf's badly calibrated views — but Shelf numbers
   carry the calibration caveat.
2. **The baseline still trails root-aligned DLT on the macro leaderboard.**
   Best combined direct (mixed, 128.05 mm) is worse than the root-aligned
   unweighted-DLT mean (122.37 mm) but better than direct unweighted DLT
   (134.43 mm) and conf-DLT (132.29 mm). On Campus alone the learned model's
   root-aligned number (118.17–118.32 mm) does edge out root-aligned DLT
   (120.61 mm). Direct-vs-root-aligned comparison matters because the learned
   model also reduces absolute translation error, not just pose shape.
3. **Cross-view features help slightly**, as in the paper: Campus cross-view
   132.34 vs per-view 132.63 mm direct. The gap is small here because the
   feature vector is only detection statistics (no backbone features).
4. Training saturates within ~4 epochs on ~1k frames — the protocol is data-
   limited, matching the leaderboard's note that Shelf/Campus alone is too
   small for the paper's standard training protocol (which pretrains on
   H36M). Long-run numbers are therefore a lower bound for the method, not
   an architecture verdict.

## Evidence

| Run | Log | Checkpoint | Config/history |
|-----|-----|------------|----------------|
| Mixed Shelf+Campus (main) | `outputs/iskakov_learnable_tri_detected.log` | `outputs/iskakov_learnable_tri_detected.pth` | `outputs/iskakov_learnable_tri_detected.config.json` |
| Campus-only (cross-view) | `outputs/iskakov_learnable_tri_campus_only.log` | `outputs/iskakov_learnable_tri_campus_only.pth` | `outputs/iskakov_learnable_tri_campus_only.config.json` |
| Campus-only per-view ablation | `outputs/iskakov_learnable_tri_campus_perview_ablation.log` | `outputs/iskakov_learnable_tri_campus_perview_ablation.pth` | `outputs/iskakov_learnable_tri_campus_perview_ablation.config.json` |

Reproduce:

```bash
# mixed (main)
CUDA_VISIBLE_DEVICES=0 /d/anaconda3/python.exe experiments/train_iskakov_baseline_shelf_campus.py \
    --datasets shelf+campus --epochs 30 --log_path outputs/iskakov_learnable_tri_detected.log
# campus-only
CUDA_VISIBLE_DEVICES=0 /d/anaconda3/python.exe experiments/train_iskakov_baseline_shelf_campus.py \
    --datasets campus --epochs 30 --log_path outputs/iskakov_learnable_tri_campus_only.log
```

## Updated leaderboard context (true-GT, 2026-08-10)

| Method | Best val MPJPE (mm) | Metric |
|---|---:|---|
| DLT baseline (root-aligned, macro mean) | 122.37 | root |
| DLT baseline (direct MJE, macro mean) | 134.43 | direct |
| **Iskakov-style learned weights (mixed, this run)** | **128.05** | direct |
| v80 (3-epoch smoke) | 408.58 | mixed |
| v57 (3-epoch smoke) | 424.63 | mixed |
| v25 (3-epoch smoke) | 430.67 | mixed |

(The learned baseline is the first method on this protocol to beat a frozen
DLT variant head-to-head on direct MPJPE; it does not yet beat the
root-aligned DLT macro mean.)
