# CVPR 2027 Roadmap: MotionFlow-MultiView

> **Status:** data-foundation repair in progress.  
> **Target venue:** CVPR 2027 (~November 2026, ~13 weeks remaining).  
> **Paper pivot:** from absolute MPJPE records to sparse-view / cross-domain robustness.

---

## 1. Executive summary

The H36M labels in `data/h36m_hf/*_multiview.npz` are circular: they are the unweighted DLT triangulation of the input 2D keypoints (`direct MJE = 0.0000 mm`).  All v25–v79 numbers on H36M therefore measure how closely a model reproduces `triangulate_dlt(p2d, cameras)`, not pose accuracy.

Current ground truth outside the circular H36M pipeline:

- **MPI-INF-3DHP** has true 3D (`univ_annot3`).  A non-circular DLT baseline on the standard protocol is **~23.8 mm MPJPE**.
- **Shelf / Campus** have true 3D.  With detection-2D inputs the DLT baseline is **~10.2 / 10.9 mm PA-MPJPE**.
- **v25 smoke on MPI-only true GT** gives **~26.15 mm MPJPE**, worse than DLT, confirming that the old leaderboard was ranking DLT mimicry, not 3D pose quality.

**Pivot.**  We abandon the "lowest MPJPE record" narrative and position the paper around **sparse-view and cross-domain robustness**: a calibrated, geometry-first fusion module that works when cameras are few, noisy, or from another domain.  CVPR 2027 is reachable only if the data foundation is fixed or cleanly pivoted within the next two weeks.

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
| **Human3.6M** | Pending original mocap world coordinates | Detected 2D (to be obtained) | Labels in `data/h36m_hf` are circular DLT(p2d, cams) | S1,S5,S6,S7,S8 train → S9,S11 test; 17 joints, 4 views |
| **MPI-INF-3DHP** | Yes (`univ_annot3`) | Current: GT-projected 2D; Future: detected 2D (OpenPose/HRNet) | Non-circular DLT baseline ~23.8 mm | Train S1/S3, validate S2/Seq1; 28 joints, 14 views |
| **Shelf** | Yes | Detection 2D (CPN/MARS) | DLT PA-MPJPE ~10.2 mm | Standard Shelf protocol |
| **Campus** | Yes | Detection 2D (CPN/MARS) | DLT PA-MPJPE ~10.9 mm | Standard Campus protocol |

**Action:** regenerate all `.npz` files from true 3D or official detections; never reuse the circular `webbridge_loader.py:182` triangulation as a label.

---

## 6. Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **True H36M 3D GT unavailable** | Cannot use H36M for model selection | Pivot fully to MPI-INF-3DHP + Shelf/Campus; use H36M only as a cross-domain test if labels ever arrive |
| **v25 still below DLT on true GT** | Loses the accuracy anchor | Make robustness the primary claim; treat v25 as a strong-but-not-record baseline |
| **SOTA comparisons too slow to run** | Related-work table weak | Use published numbers on identical splits when possible; run only the two most relevant baselines |
| **Cross-dataset transfer fails** | Cross-domain contribution collapses | Add a small domain-adaptation wrapper or per-dataset batch-normalization; keep the core model domain-agnostic |
| **Timeline compression** | Cannot finish all six weeks | Drop Shelf/Campus full runs and keep only MPI main + H36M/Shelf transfer if labels arrive late |

---

## 7. Immediate next actions

1. **Confirm H36M true-GT access.**  Search local and A800-D storage for original `.mat`/`.cdf` mocap files; if absent, file a formal data request or pivot immediately.
2. **Regenerate MPI `.npz` with detected 2D.**  Use OpenPose/HRNet detections and the true `univ_annot3` labels; retire the GT-2D circular variant.
3. **Run the corrected DLT/v25/v46/v57 baselines on true MPI GT.**  Document `MPJPE`, `PA-MPJPE`, `PCK@50/100/150`, and `AUC`.
4. **Draft the sparse-view evaluation harness.**  Reuse `eval_variable_views.py` to produce `MPJPE@k` curves for `k = 2 … 14` and for `k = 2 … 4` on H36M.
5. **Lock the paper story by end of week 2.**  The contribution is robustness; accuracy numbers are supporting evidence, not the headline.

---

## 8. One-paragraph paper story

*Multi-view 3D pose estimation has been chasing circular Human3.6M labels, rewarding networks that simply learn to reproduce DLT.  We show that once the labels are true world mocap or detected-2D triangulations, the problem becomes harder and the real differentiator is robustness: to few views, noisy detections, and cross-domain capture setups.  We present a geometry-first fusion module, evaluate it with honest baselines on MPI-INF-3DHP and Shelf/Campus, and demonstrate sparse-view and cross-domain curves that existing leaderboard records obscured.*
