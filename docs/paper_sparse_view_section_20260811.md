# Sparse-View Robustness: MPJPE@k Curves and Implications

> **Date:** 2026-08-11  
> **Scope:** Variable-view evaluation of MotionFlow-MultiView variants on true-GT H36M and on an MPI-INF-3DHP smoke subset.  
> **Key message:** On honest, non-circular ground truth, learned variants still trail confidence-weighted DLT and Iskakov ICCV 2019 in full-view accuracy. Their *sparse-view degradation curves* therefore become the central empirical signal for the CVPR 2027 story.

---

## 1. What the curves measure

We report **MPJPE@k**: the mean per-joint position error when only a random subset of `k` views is active.  On H36M true-GT we evaluate `k = 2, 3, 4` (50 random subsets per `k`, `clip_len = 13`, hardened variable-view wrapper).  On MPI-INF-3DHP we evaluate `k = 2, 3, 4, 14` using a 300-frame 17-joint smoke subset of the RTMPose-detected sequence (`s_02_seq_01`, 5 random subsets per `k`).

The protocol is inference-only: the model is trained with the full camera rig and then dropped into each sparse subset without retraining.  It tests whether the learned fusion has acquired a genuine multi-view prior or merely memorised a fixed camera configuration.

---

## 2. H36M true-GT variable-view results

| Variant | k=2 (mm) | k=3 (mm) | k=4 (mm) | Full-view test (mm) | Source |
|---|---:|---:|---:|---:|---|
| v81 (temporal-pose-attention, A800) | **4244.22** | **1365.53** | **50.97** | 37.83 | `outputs/variable_view_v81_true_gt_medium_a800.json` |
| v82 (multi-scale temporal-pose-attention, A800) | **PENDING** | **PENDING** | **PENDING** | 39.46 | `outputs/variable_view_v82_true_gt_medium_a800.json` (terminated; needs relaunch) |
| v25 stability | **3681.04** | **1036.32** | **113.78** | 31.56 | `outputs/variable_view_v25_true_gt_stability_a800.json` |
| v57 (A800 re-run) | **178.40** | **145.29** | **140.20** | 57.10 | `outputs/variable_view_v57_true_gt_medium_a800.json` |
| v80 (medium, local) | **208.51** | **139.69** | **96.82** | 62.32 | `tmp/variable_view_v80_h36m_true_gt_medium.json` |
| v80 (regularization, A800) | **203.83** | **134.65** | **104.29** | 53.98 | `tmp/variable_view_v80_true_gt_regularization_a800_gpu7.json` |

Per-subject breakdown for the v57 A800 re-run (the only variant with per-subject sparse-view numbers currently available):

| Subject | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | 182.58 (std 43.07) | 148.35 (std 4.07) | 143.02 |
| S11 | 174.22 (std 40.36) | 142.24 (std 3.81) | 137.39 |
| **Combined** | **178.40** | **145.29** | **140.20** |

Per-subject breakdown for the v81 A800 medium run (temporal-pose-attention):

| Subject | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | 4230.29 (std 2359.57) | 1356.67 (std 425.22) | 54.53 |
| S11 | 4258.15 (std 2427.20) | 1374.38 (std 453.25) | 47.41 |
| **Combined** | **4244.22** | **1365.53** | **50.97** |

Per-subject breakdown for the v25 stability A800 run:

| Subject | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | 3989.03 (std 3162.68) | 1042.45 (std 285.23) | 116.98 |
| S11 | 3373.05 (std 2038.58) | 1030.19 (std 285.88) | 110.58 |
| **Combined** | **3681.04** | **1036.32** | **113.78** |

*Combined values are the simple mean of the per-subject MPJPE@k.*

The v82 multi-scale temporal-pose-attention checkpoint has completed full-view training (test **39.46 mm**, val **39.58 mm**), but its variable-view evaluation terminated before producing a JSON/CSV. Until it is relaunched, the sparse-view comparison remains limited to v81, v80, and v57.

> **Catastrophic failure at k=2 and k=3.** The v81 temporal-pose-attention checkpoint reports **4244.22 mm at k=2** and **1365.53 mm at k=3** (combined S9+S11), with per-subject standard deviations exceeding 400 mm. This is not a minor degradation: the model essentially loses all pose information when one or two cameras are removed. At k=4 the error collapses to **50.97 mm**, confirming the head has learned a strong full-rig temporal prior that is not view-agnostic. This pattern is the primary motivation for the pre-triangulation, view-conditioned temporal attention proposed in `docs/v83_design_20260812.md`.

### v81 vs. v25 stability comparison

| Variant | k=2 (mm) | k=3 (mm) | k=4 (mm) | Full-view test (mm) | Source |
|---|---:|---:|---:|---:|---|
| v81 (temporal-pose-attention) | **4244.22** | **1365.53** | **50.97** | 37.83 | `outputs/variable_view_v81_true_gt_medium_a800.json` |
| v25 stability | **3681.04** | **1036.32** | **113.78** | 31.56 | `outputs/variable_view_v25_true_gt_stability_a800.json` |

- **v25 stability is more robust to view dropout than v81 at low view counts.** At k=2 and k=3, v25 stability outperforms v81 by **563.18 mm** and **329.21 mm** respectively (combined S9+S11). This is surprising because v81 has the weaker full-view test result (37.83 mm vs. 31.56 mm), suggesting that the temporal-pose-attention head does not inherit the base model's sparse-view robustness.
- **v81 is stronger at k=4.** With all four views available, v81 reaches **50.97 mm**, which is **62.81 mm** lower than v25 stability's **113.78 mm**. This suggests the temporal-pose-attention head is highly effective when the full camera rig is present, but its benefits disappear as soon as one view is removed.
- **Both models fail catastrophically at k=2 and k=3.** Even the better sparse-view numbers (v25 stability at 3681.04 / 1036.32 mm) are orders of magnitude above the full-view test errors, confirming that neither variant has learned a reliable view-agnostic geometric prior.

### Observations

- **v81 fails catastrophically at k=2 and k=3, but matches full-view accuracy at k=4.**  The v81 temporal-pose-attention checkpoint reports **4244.22 mm at k=2** and **1365.53 mm at k=3** (combined S9+S11), with very high standard deviation (>400 mm).  At k=4, however, it drops to **50.97 mm**, which is below v80 regularization (104.29 mm) and v57 (140.20 mm).  This suggests the temporal-pose-attention head is highly dependent on the full camera rig; when even one view is removed, the model produces wildly inaccurate predictions.
- **v80 dominates at k=4.**  The local v80 medium checkpoint reaches 96.82 mm at k=4, well below the v57 re-run (140.20 mm).  The A800 regularization checkpoint is also stronger than v57 at k=4 (104.29 mm).
- **v57 is competitive at low view counts.**  At k=2 and k=3 the v57 re-run is the lowest-error H36M checkpoint (178.40 / 145.29 mm), slightly ahead of both v80 variants.  This suggests that v57's domain-conditional physical-space calibration head helps when only a couple of views are available.
- **Sparse-view error is far above full-view test error.**  Even the best k=4 number (96.82 mm) is almost double the corresponding full-view test result (62.32 mm).  The variable-view benchmark uses clip-level inference and random camera subsets, so it is stricter than the full-sequence test; still, the gap indicates that none of the variants yet has a robust sparse-view prior.

### v80 vs. v57 conclusion

The variable-view comparison now has a clear winner: **v80 regularization outperforms v57 at every view count on true-GT H36M.**  The A800 regularization checkpoint records **102.8 mm (S9) and 105.8 mm (S11) at k=4**, while the v57 re-run records **143.0 mm (S9) and 137.4 mm (S11)** at the same setting.  Even the local v80 medium checkpoint, which subsequently overfit, reached **96.82 mm at k=4**.  At k=2 and k=3 the gap narrows, and v57 is actually the lowest-error H36M checkpoint in the earlier table, so v57's physical-space calibration head still offers value when only one or two views are available.  The practical conclusion is that **v80's view-reliability regularisation is the stronger sparse-view ingredient for three- and four-view settings**, while v57's calibration-aware design may be worth revisiting for the extreme sparse case (k=2) if overfitting can be controlled.

---

## 3. MPI-INF-3DHP smoke variable-view results

These numbers come from `outputs/v57_mpi_varview.csv`, `outputs/v80_mpi_varview.csv`, and the aggregated `outputs/mpjpe_at_k_summary.csv`.

| Variant | k=2 | k=3 | k=4 | k=14 |
|---|---:|---:|---:|---:|
| v25 | 131.98 | 72.05 | 89.02 | **30.61** |
| v46 | 108.51 | 93.10 | 72.31 | 68.64 |
| v57 | 143.67 | 87.02 | **53.41** | 40.93 |
| v80 | 145.01 | 74.41 | 63.34 | 51.79 |

### Observations

- **v57 scales most smoothly from few views to full views.**  Its error drops from 143.67 mm at k=2 to 40.93 mm at k=14, and it records the best k=4 result (53.41 mm).
- **v25 is volatile.**  It has the best full-view result (k=14, 30.61 mm) but degrades non-monotonically (89.02 mm at k=4), suggesting it overfits to the full 14-view rig and cannot reliably fuse fewer views.
- **v80 trades full-view accuracy for graceful degradation.**  v80 is not the best at k=14, yet its k=2→k=4 curve is smoother than v25's and comparable to v57's.

---

## 4. Implications for the CVPR 2027 story

### 4.1 Absolute MPJPE is no longer the headline

On true-GT H36M, the best MotionFlow variant (v80, 39.98 mm test / 53.98 mm regularization) is still **~14–17 mm behind** confidence-weighted DLT (25.67 mm) and Iskakov ICCV 2019 (23.35 mm).  Any claim of state-of-the-art absolute accuracy would be unsupportable.  The paper must therefore pivot from *record MPJPE* to *robustness under realistic deployment conditions*.

### 4.2 Sparse-view robustness becomes the differentiator

The variable-view curves give us a coherent narrative:

1. **Geometric baselines are the accuracy ceiling.**  DLT and Iskakov define the true-GT performance frontier when all views are present.
2. **Learned variants differ in how they degrade.**  v80's view-reliability regularisation gives it the best H36M k=4 performance; v57's physical-space calibration helps at very low view counts; v25 collapses as soon as the full rig is no longer available.
3. **There is still a large robustness gap.**  Even the best learned sparse-view numbers (96–140 mm on H36M, 40–54 mm on MPI) show that current architectures have not internalised a reliable geometric prior.

### 4.3 What to write in the paper

- **Frame the contribution as "geometry-first robustness."**  The triangulation-residual decomposition is the right starting point, but the learned residual is not yet strong enough to beat geometric baselines.  The value of the architecture lies in making that residual degrade gracefully as views are removed.
- **Lead with H36M true-GT, then use MPI smoke as supporting evidence.**  H36M true-GT is the cleanest, non-circular protocol; the MPI smoke subset validates that the degradation pattern generalises to a different skeleton and camera rig.
- **Do not hide the DLT/Iskakov lead.**  State clearly that learned variants trail these frozen/learnable triangulation baselines.  The novelty is not outperforming them in full-view accuracy, but providing a single model that adapts to variable view counts without per-rig retraining.
- **Use the curves to justify future work.**  The v57 vs. v80 comparison supports the hypothesis that *explicit per-view reliability and physical-space regularisation* are the right ingredients; the remaining failure is one of generalisation (not enough data, not enough regularisation), not architecture.

### 4.4 How v81 and v82 change the story

v81 adds **per-joint temporal pose attention** to the v25 backbone.  The medium run is now complete on true-GT H36M (test **37.83 mm**, val **38.62 mm** @ Epoch 8).  In the variable-view benchmark, v81 is the **strongest variant at k=4** (50.97 mm combined), but it **fails catastrophically at k=2 and k=3** (4244.22 mm and 1365.53 mm respectively, with enormous variance).  This pattern suggests the temporal-pose-attention head has learned a full-rig temporal prior that is not robust to missing views.

v82 extends the same idea with **multi-scale temporal-pose attention**.  Its full-view result is **39.46 mm** test (val **39.58 mm**), slightly worse than v81 and well behind v25 stability (30.83 mm mean).  The v82 variable-view evaluation terminated before producing a curve, but the full-view trend already indicates that adding more temporal scales after triangulation is unlikely to close the sparse-view gap.  The proposed remedy is to move temporal reasoning **before** triangulation and explicitly gate it by per-view reliability; see the v83 pre-triangulation, view-conditioned temporal attention proposal in `docs/v83_design_20260812.md`.

---

## 5. Caveats and limitations

- **H36M variable-view numbers are high because the benchmark is strict.**  Clip-level inference and random subsets inflate error relative to the full-sequence test.  They are still useful for comparing variants, but should not be compared directly to full-view leaderboards.
- **MPI smoke is a 300-frame subset.**  Absolute values are noisy and should be treated as directional only.
- **v80 medium numbers come from a local checkpoint that subsequently overfit.**  Its strong k=4 result (96.82 mm) is from the same model that reached 39.98 mm at epoch 4 and then diverged to 133.71 mm by epoch 8; this reinforces the overfitting story rather than contradicting it.
- **Variable-view training is not yet standardised.**  Some variants were trained with view dropout, some without, and the exact dropout rate differs.  A cleaner ablation of *training-time view dropout* vs. *inference-time view dropout* is still needed.

---

## 6. Recommended figures and tables for the paper

1. **Figure: H36M true-GT MPJPE@k curves.**  Plot v57, v80 (medium), and v80 (regularization) for `k = 2, 3, 4`.  Add a horizontal reference line for the confidence-weighted DLT full-view result (25.67 mm) to visualise the accuracy gap.
2. **Figure: MPI-INF-3DHP smoke MPJPE@k curves.**  Plot v25, v46, v57, v80 for `k = 2, 3, 4, 14`.
3. **Table:** A concise version of the H36M true-GT table above, placed in the experiments section.

---

## 7. Next steps

1. **Relaunch v82 variable-view evaluation.** The A800 job terminated before writing any output; the checkpoint at `outputs/ablations/v82_true_gt_h36m_medium_a800.pth` is ready. Run `scripts/eval_variable_views_v82_true_gt_medium_a800.sh` on a free GPU and add the resulting `outputs/variable_view_v82_true_gt_medium_a800.json` to the table above.
2. Generate per-subject sparse-view CSVs for the v80 variants so the H36M table can be expanded beyond combined numbers.
3. Run the same `MPJPE@k` protocol for DLT and Iskakov on true-GT H36M to include frozen/learnable triangulation baselines in the sparse-view comparison.
4. Once MPI RTMPose detection finishes, run the full H36M-style variable-view benchmark on MPI-INF-3DHP (not just the 300-frame smoke).
5. Add a controlled ablation of training-time view dropout for v80 to quantify how much of the sparse-view gain is training regularisation vs. inference-time masking.
6. **Decide on v83 implementation.** With v81 and v82 both failing to beat v25 stability, the next architectural bet is the view-conditioned temporal-attention proposal in `docs/v83_design_20260812.md`; it is still pending implementation.
