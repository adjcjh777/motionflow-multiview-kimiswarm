# CVPR 2027 Roadmap: MotionFlow-MultiView

> **Status:** data-foundation repair in progress.  
> **Target venue:** CVPR 2027 (~November 2026, ~13 weeks remaining).  
> **Paper pivot:** from absolute MPJPE records to sparse-view / cross-domain robustness.

---

## 1. Executive summary

The H36M labels in `data/h36m_hf/*_multiview.npz` are circular: they are the unweighted DLT triangulation of the input 2D keypoints (`direct MJE = 0.0000 mm`).  All v25–v79 numbers on H36M therefore measure how closely a model reproduces `triangulate_dlt(p2d, cameras)`, not pose accuracy.

Current ground truth outside the circular H36M pipeline:

- **Human3.6M true GT** is now available in `data/h36m_true_gt/`. The standard
  protocol is S1,S5,S6,S7,S8 → S9/S11; DLT (conf-weighted) is 25.87 mm and
  Iskakov ICCV 2019 reaches 23.35 mm combined direct (`docs/results_true_gt_h36m.md`).
- **MPI-INF-3DHP** has true 3D (`univ_annot3`).  A non-circular DLT baseline on the standard protocol is **~23.8 mm MPJPE**.
- **Shelf / Campus** have true 3D.  On the rebuilt detected-2D `.npz`
  (`data/webbridge/shelf_campus_detected/`, verified non-circular on
  2026-08-10), the DLT baseline is **134.43 mm direct MJE / 122.37 mm
  root-aligned MPJPE** (mean of Shelf+Campus val); see
  `docs/results_true_gt_shelf_campus.md`.  The older "~10.2 / 10.9 mm
  PA-MPJPE" figures came from the superseded GT-projection protocol and must
  not be cited.
- **v25 smoke on MPI-only true GT** gives **~26.15 mm MPJPE**, worse than DLT, confirming that the old leaderboard was ranking DLT mimicry, not 3D pose quality.

**Pivot.**  We abandon the "lowest MPJPE record" narrative and position the paper around **sparse-view and cross-domain robustness**: a calibrated, geometry-first fusion module that works when cameras are few, noisy, or from another domain.  CVPR 2027 is reachable only if the data foundation is fixed or cleanly pivoted within the next two weeks.

### Status update (2026-08-11)

- **True-GT Shelf/Campus leaderboard complete**
  (`docs/results_true_gt_shelf_campus.md`): DLT 122.37/134.43 mm vs
  v80 **408.58 mm**, v57 **424.63 mm**, v25 **430.67 mm** (3-epoch smoke, all
  runs stable). Learned-model ranking: **v80 (view-reliability weighting) >
  v57 > v25**; all learned models are far undertrained relative to DLT.
- **Reprojection audit** (`scripts/check_true_gt_reprojection.py`): Campus is
  camera/2D/3D-consistent (~7.7 px RMSE) but Shelf is systematically
  misaligned (~53.7 px RMSE, 87/87 val frames), so **Campus (3 views) is the
  primary sparse-view benchmark**; Shelf numbers need a calibration caveat.
- **H36M true-GT leaderboard live** (`docs/results_true_gt_h36m.md`):
  true mocap world coordinates are in `data/h36m_true_gt/`, the standard
  protocol is S1,S5,S6,S7,S8 → S9/S11, DLT (conf-weighted) = 25.87 mm,
  Iskakov ICCV 2019 = 23.35 mm (current leader), v80 best converged =
  42.60 mm (local v3, 2 epochs; A800 v2 best 39.70 mm; local medium
  39.98 mm at epoch 4), and v25 medium
  finished at 72.80 mm best epoch 2 before diverging to 207.62 mm. The
  old circular-label 0.62 mm result is superseded.
- **`data/webbridge/h36m_corrected/` verified circular as well**
  (direct MJE 0.0000 mm); the "corrected" suffix refers to cameras, not labels.
- **MPI-INF-3DHP detected-2D still blocked**: local `raw/S*/Seq*/` contains
  only `annot.mat` + `camera.calibration`, no `imageSequence/`; the
  GT+noise fallback is not a substitute for real detections.

---

## 2. Six-week plan (post data-foundation repair)

Adapted from the project CVPR 2027 phases, compressed into a single six-week sprint.

| Week | Phase | Deliverables | Exit criteria |
|---|---|---|---|
| **1** | Fix data foundation | Decide H36M true-GT source or pivot to MPI/Shelf; regenerate canonical `.npz`; run DLT diagnostic | Non-zero DLT-to-label MPJPE on every benchmark dataset |
| **2** | Rebuild baselines | Run DLT, v25, v46, v57 on the corrected protocol | Stable rank order; v25 baseline on true MPI GT reproduced |
| **3** | Add standard SOTA | Implement or wrap Iskakov (learnable triangulation), VoxelPose, and a lightweight MVPose baseline | Reported MPJPE/PA-MPJPE on same split |
| **4** | Robustness & cross-dataset | Sparse-view `MPJPE@k` curves, 2D noise / occlusion / view-dropout, calibration-perturbation matrix, cross-dataset transfer H36M↔MPI | Figures and tables ready |
| **5** | Rewrite paper | Real citations, corrected main table, robustness story, method section | Full draft internally reviewable |
| **6** | Submission buffer | Supplementary video, final numbers, camera-ready checks | CVPR 2027 submission package |

---

## 3. Revised paper contribution

1. **Sparse-view robustness.**  Reliable 3D pose from as few as 2–3 views, quantified by `MPJPE@k` curves rather than a single best-view number.
2. **Cross-domain generalization.**  A single model trained on one dataset (or mixed data) transfers to another without per-dataset pose heads, using domain-agnostic ray features.
3. **Calibration-robust fusion.**  Principled handling of noise, occlusion, and mild calibration drift through geometry-first weighted triangulation and residual refinement.
4. **Honest, non-circular benchmarking.**  We re-derive all labels from official mocap or detected 2D and report baselines that actually measure pose accuracy.
5. **Efficient plug-in module.**  The fusion block remains compact, warm-startable, and packaged as a `MultiViewFusionPlugin` inside MotionFlow.

---

## 4. Minimal model lineup for re-ranking

| Model | Role in the paper | Expected status after data fix |
|---|---|---|
| **DLT** | Geometric baseline; the non-learning lower bound | Must be run on true GT to set the real baseline |
| **v25** | Best previous geometry-fusion anchor | Re-evaluate; likely starts above DLT on true MPI GT |
| **v46 (SVG)** | Sparse-view graph / variable-view baseline | Run `eval_variable_views.py` with `MPJPE@k` |
| **v57** | Latest robustness/calibration-aware fusion | Include if it shows calibration or cross-domain improvement |
| **Proposed: Sparse-view adaptive selector** | New lightweight module: budgeted top-k view selection + visibility gating | Design and smoke-test in weeks 2–3 |

The leaderboard question is no longer "who has the smallest MPJPE?" but "who degrades least when views, labels, or domains change?"

---

## 5. Dataset protocol table

| Dataset | True 3D? | 2D input | Current status | Protocol |
|---|---|---|---|---|
| **Human3.6M** | Yes (`data/h36m_true_gt/`) | Detected 2D supplied with true-GT release | True mocap world coordinates available and non-circular; old `data/h36m_hf` and `data/webbridge/h36m*.npz` are circular DLT(p2d, cams) and must not be used | S1,S5,S6,S7,S8 train → S9,S11 test; 17 joints, 4 views |
| **MPI-INF-3DHP** | Yes (`univ_annot3`) | Current: GT-projected 2D; Future: detected 2D (blocked on missing `imageSequence`) | Non-circular DLT baseline ~23.8 mm | Train S1/S3, validate S2/Seq1; 28 joints, 14 views |
| **Shelf** | Yes | Detection 2D | Non-circular `.npz` rebuilt; DLT val 130.77 mm direct / 124.13 mm root-aligned; calibration coarse (53.7 px reproj RMSE) — report with caveat | Standard Shelf protocol, 5 views |
| **Campus** | Yes | Detection 2D | Non-circular `.npz` rebuilt; DLT val 138.08 mm direct / 120.61 mm root-aligned; cameras consistent (7.7 px reproj RMSE) — primary sparse-view benchmark | Standard Campus protocol, 3 views |

**Action:** regenerate all `.npz` files from true 3D or official detections; never reuse the circular `webbridge_loader.py:182` triangulation as a label.

---

## 6. Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **v25 diverges after epoch 2 on true GT** | Learned models cannot yet beat geometric baselines | Diagnose overfitting (LR, auxiliary losses, augmentation); add regularisation / mixed-dataset training; run v80/v57 medium |
| **v25 still below DLT on true GT** | Loses the accuracy anchor | Make robustness the primary claim; treat v25 as a strong-but-not-record baseline |
| **SOTA comparisons too slow to run** | Related-work table weak | Use published numbers on identical splits when possible; run only the two most relevant baselines |
| **Cross-dataset transfer fails** | Cross-domain contribution collapses | Add a small domain-adaptation wrapper or per-dataset batch-normalization; keep the core model domain-agnostic |
| **Timeline compression** | Cannot finish all six weeks | Drop Shelf/Campus full runs and keep only MPI main + H36M/Shelf transfer if labels arrive late |

---

## 7. Immediate next actions

1. **Diagnose v25 divergence on H36M true GT and run v80/v57 medium.** The
   true mocap `.npz` are already in `data/h36m_true_gt/`. v25 completed
   8 epochs but diverged after epoch 2 (72.80 → 207.62 mm). Debug
   overfitting, then run v80 and v57 medium to fill
   `docs/results_true_gt_h36m.md`.
2. **Obtain MPI `imageSequence` (external).** Then run
   `scripts/generate_mpi_detected_2d.py --detector auto` (MediaPipe/OpenPose
   wrappers are already implemented) to replace the GT+noise fallback.
3. **Longer true-GT training on Campus.** Extend v80/v57 from the 3-epoch
   smoke to full-data, >= 20-epoch runs on
   `configs/splits/shelf_campus_detected_smoke.yaml` and check whether any
   learned model closes the gap to the ~122 mm root-aligned DLT baseline.
4. **Draft the sparse-view evaluation harness.** Reuse `eval_variable_views.py`
   to produce `MPJPE@k` curves for `k = 2 … 14` (MPI) and `k = 2 … 4` (H36M,
   once true GT arrives).
5. **Lock the paper story by end of week 2.** The contribution is robustness;
   accuracy numbers are supporting evidence, not the headline. Story is
   anchored in this roadmap + `docs/results_true_gt_shelf_campus.md`; all
   superseded drafts are banner-marked.

---

## 8. One-paragraph paper story

*Multi-view 3D pose estimation has been chasing circular Human3.6M labels, rewarding networks that simply learn to reproduce DLT.  We show that once the labels are true world mocap or detected-2D triangulations, the problem becomes harder and the real differentiator is robustness: to few views, noisy detections, and cross-domain capture setups.  We present a geometry-first fusion module, evaluate it with honest baselines on MPI-INF-3DHP and Shelf/Campus, and demonstrate sparse-view and cross-domain curves that existing leaderboard records obscured.*
