# v2 / v3 MPI-INF-3DHP Baseline Reproduction — T17

**Task:** T17 of Swarm Iteration 20 v4 (#76)  
**Goal:** Re-run or collect existing v2 and v3 MPI-INF-3DHP clean / robustness / variable-view evaluations, summarise them in a single table, and identify the strongest baseline that v4 must beat.  
**Date:** 2026-08-07  
**Branch:** `feat/swarm-iter20-v4`

---

## 1. Summary

This document consolidates every available MPI-INF-3DHP baseline result for the two major architecture generations before v4:

* **v2 family**: `OmniMultiViewFusionV2`, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, and `BayesianTriV2` variants.
* **v3 family**: `OmniMultiViewFusionV3` (hierarchical multi-scale fusion + camera-conditioned epipolar bias).

All numbers are reported on **MPI-INF-3DHP S2/Seq1 (14 views, 28 joints)** unless otherwise noted.

**Bottom line:** The strongest verified single-model baseline is **Bayesian Tri v2 stabilized (9.03 mm MPJPE / 5.69 mm PA-MPJPE)**. The strongest ensemble is **Bayesian Tri v2 ensemble (8.61 mm MPJPE / 5.38 mm PA-MPJPE)**. v4 must beat **9.03 mm MPJPE** on clean S2/Seq1 to claim a single-model improvement over the best v2 baseline.

---

## 2. Clean MPI-INF-3DHP S2/Seq1 results

| Model / Run | Checkpoint / Source | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---|---:|---:|---|
| Cross-view residual + PP (d=32, h=64, small) | `docs/results_icra_cvpr_2027.md` | 10.25 | 8.60 | No PP correction baseline |
| Cross-view residual + PP (d=64, h=128, full, 20 ep) | `docs/results_icra_cvpr_2027.md` | 9.32 | 5.37 | Full model, best single non-Bayesian |
| Cross-view residual + PP full (re-eval) | `outputs/crossview_pp_full_ppw005_20ep_eval.json` | **9.32** | **5.37** | Confirmed reproduction |
| Bayesian Tri v2 (large scale) | `outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_eval.json` | 9.71 | 5.79 | d=128 single |
| Bayesian Tri v2 (PP full) | `outputs/eval_bayesian_tri_pp_full_mpiinf3dhp.log` | 9.81 | 5.84 | — |
| **Bayesian Tri v2 stabilized** | `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp_eval.json` | **9.03** | **5.69** | **Best single-model v2** |
| **Bayesian Tri v2 ensemble (stabilized + aug, epoch 22)** | `docs/results_icra_cvpr_2027.md` | **8.61** | **5.38** | **Best overall v2** |
| v2 dense+graph (d=128, freeze+end-to-end) | `outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.json` | 25.21 | 24.14 | Still training / not converged |
| v3 smoke (d=32, synthetic 4-view) | `outputs/eval_omniview_fusion_v3_smoke.json` | 125.31 | 12.50 | Smoke only, not representative |
| v3 full MPI (training in progress) | `docs/swarm_iter19_status.md` | 25.72 (val) | — | Epoch 1; full eval pending |

**Winner:** Bayesian Tri v2 stabilized is the strongest *single-model* baseline. The ensemble is the strongest *overall* baseline.

---

## 3. Calibration-robustness matrix (best v2 full model)

Condition matrix for the cross-view residual + PP full model (`outputs/crossview_pp_full_ppw005_20ep_eval.json` / `docs/tables/icra2027/robustness.md`).

| Condition | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| clean | 9.32 | 5.37 | 1.000 | 1.000 | 1.000 | 0.938 |
| rot_0.5_deg | 17.24 | 8.03 | 0.996 | 1.000 | 1.000 | 0.885 |
| rot_1.0_deg | 27.03 | 13.79 | 0.938 | 1.000 | 1.000 | 0.820 |
| trans_5mm | 9.74 | 5.41 | 1.000 | 1.000 | 1.000 | 0.935 |
| trans_10mm | 11.04 | 5.91 | 1.000 | 1.000 | 1.000 | 0.926 |
| focal_1pct | 18.01 | 9.02 | 1.000 | 1.000 | 1.000 | 0.880 |
| focal_2pct | 29.42 | 13.13 | 0.989 | 1.000 | 1.000 | 0.804 |
| cxcy_3px | 11.18 | 6.05 | 1.000 | 1.000 | 1.000 | 0.925 |
| cxcy_5px | 13.78 | 6.76 | 1.000 | 1.000 | 1.000 | 0.908 |

The same conditions are also documented in `docs/results_icra_cvpr_2027.md` and `docs/tables/icra2027/robustness.md`.

---

## 4. Extended robustness — Bayesian Tri v2 stabilized

From `docs/results_icra_cvpr_2027.md` and `outputs/extended_robustness_matrix_bayesian_tri_v2_stabilized/robustness_matrix.json`.

| Condition | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| clean | 9.03 | 5.69 | 1.000 | 1.000 | 1.000 | 0.940 |
| noise_1.0px | 9.10 | 5.83 | 1.000 | 1.000 | 1.000 | 0.939 |
| joint_occlusion_20 | 14.56 | 12.90 | 0.999 | 1.000 | 1.000 | 0.903 |
| view_dropout_30 | 18.15 | 7.02 | 0.995 | 1.000 | 1.000 | 0.879 |
| joint_occlusion_20 + view_dropout_30 | 22.02 | 15.45 | 0.964 | 1.000 | 1.000 | 0.853 |

(Full 14-condition table is in `docs/results_icra_cvpr_2027.md`.)

---

## 5. Variable-view MPJPE@k (k = 2…14)

### 5.1 Bayesian Tri v2 stabilized — strongest single-model baseline

From `outputs/variable_views_bayesian_tri_v2_stabilized.json` (10 subsets per k).

| k | MPJPE (mm) | std (mm) | subsets |
|---:|---:|---:|---:|
| 2 | 280.02 | 82.38 | 10 |
| 3 | 174.39 | 97.58 | 10 |
| 4 | 113.04 | 17.26 | 10 |
| 5 | 98.30 | 50.33 | 10 |
| 6 | 70.95 | 14.94 | 10 |
| 7 | 51.35 | 15.79 | 10 |
| 8 | 41.74 | 13.63 | 10 |
| 9 | 39.10 | 12.63 | 10 |
| 10 | 28.28 | 7.21 | 10 |
| 11 | 27.99 | 8.48 | 10 |
| 12 | 23.98 | 4.73 | 10 |
| 13 | 15.47 | 2.04 | 10 |
| 14 | 8.99 | 0.00 | 1 |

**Take-away:** 2/3 views are catastrophic for the current best single-model v2 (~280 mm / ~174 mm). This is the primary gap v4 targets.

### 5.2 Cross-view residual + PP full (20 epochs)

From `outputs/variable_views_crossview_residual_quick.json` (1 subset per k, quick sweep).

| k | MPJPE (mm) | std (mm) | subsets |
|---:|---:|---:|---:|
| 2 | 98.16 | 0.00 | 1 |
| 3 | 75.02 | 0.00 | 1 |
| 4 | 68.71 | 0.00 | 1 |
| 5 | 63.59 | 0.00 | 1 |
| 6 | 51.81 | 0.00 | 1 |
| 14 | 14.72 | 0.00 | 1 |

This cross-view residual model is more robust at low view counts than the Bayesian Tri v2 stabilized, but its full-view accuracy is worse (14.72 mm vs 8.99 mm).

### 5.3 v2 dense+graph (d=128, training)

From `outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.json`.

| k | MPJPE (mm) | std (mm) | subsets |
|---:|---:|---:|---:|
| 2 | 2179.09 | 324.67 | 10 |
| 3 | 1884.66 | 319.68 | 10 |
| 4 | 2024.17 | 327.81 | 10 |
| 14 | 25.21 | 0.00 | 1 |

This model still suffers from the catastrophic 2/3-view failure and has not yet converged.

---

## 6. v3 status

`OmniMultiViewFusionV3` extends v2 with hierarchical multi-scale fusion, camera conditioning, and epipolar-biased attention (`docs/omniview_fusion_v3_design.md`).

* **Smoke test:** `python experiments/eval_omniview_fusion_v3_mpiinf3dhp.py --smoke` passes and produces `outputs/eval_omniview_fusion_v3_smoke.json`.
* **Full MPI training:** Started on A800-D (GPU 5). As of `docs/swarm_iter19_status.md`, the model is at **Epoch 1 with val_MPJPE ≈ 25.72 mm**.
* **Full eval:** No complete clean/robustness/variable-view v3 eval is available yet because the checkpoint (`outputs/omniview_fusion_v3_mpiinf3dhp.pth`) corresponds to a small d=32 smoke model and cannot be loaded into the full 14-view configuration without mismatch. A real v3 full eval should be queued once the A800 run produces a converged checkpoint.

---

## 7. Single comparison table (≥3 conditions)

| Condition | Cross-view residual + PP full | Bayesian Tri v2 stabilized | v2 dense+graph (training) | v3 (status) |
|---|---:|---:|---:|---|
| **Clean MPJPE (mm)** | **9.32** | **9.03** | 25.21 | 25.72 val (Epoch 1) |
| **Clean PA-MPJPE (mm)** | 5.37 | 5.69 | 24.14 | — |
| **Rot 0.5° MPJPE (mm)** | 17.24 | — | 30.19 | pending |
| **Focal 1% MPJPE (mm)** | 18.01 | — | 23.92 | pending |
| **Variable-view k=2 MPJPE (mm)** | 98.16 | 280.02 | 2179.09 | pending |
| **Variable-view k=3 MPJPE (mm)** | 75.02 | 174.39 | 1884.66 | pending |
| **Variable-view k=14 MPJPE (mm)** | 14.72 | 8.99 | 25.21 | pending |

---

## 8. Strongest baseline to beat

| Category | Model | MPJPE (mm) | PA-MPJPE (mm) |
|---|---|---:|---:|
| Best single model | Bayesian Tri v2 stabilized | **9.03** | **5.69** |
| Best ensemble | Bayesian Tri v2 ensemble (stabilized + aug) | **8.61** | **5.38** |
| Best non-Bayesian single | Cross-view residual + PP full (20 ep) | 9.32 | 5.37 |

**Recommendation for v4:**

1. **Primary target:** beat **9.03 mm MPJPE** on clean MPI-INF-3DHP S2/Seq1 (single model).
2. **Stretch target:** beat **8.61 mm MPJPE** with an single model, removing the need for an Bayesian Tri ensemble.
3. **Hard variable-view target:** bring **2-view MPJPE below 50 mm** (currently 280 mm for the best single model and ~2000 mm for the dense+graph v2).

---

## 9. Files referenced

* `docs/results_icra_cvpr_2027.md`
* `docs/swarm_iter19_status.md`
* `docs/omniview_fusion_v3_design.md`
* `docs/tables/icra2027/robustness.md`
* `outputs/crossview_pp_full_ppw005_20ep_eval.json`
* `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp_eval.json`
* `outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_eval.json`
* `outputs/variable_views_bayesian_tri_v2_stabilized.json`
* `outputs/variable_views_crossview_residual_quick.json`
* `outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.json`
* `outputs/eval_omniview_fusion_v3_smoke.json`
* `outputs/omniview_fusion_v3_mpiinf3dhp.pth`

---

## 10. Verification

```bash
# Confirm v3 eval script still smoke-passes (no real checkpoint required)
python experiments/eval_omniview_fusion_v3_mpiinf3dhp.py --smoke
```

The v3 smoke test was re-run during the production of this document; it writes `outputs/eval_omniview_fusion_v3_smoke.json`.

---

## 11. Next steps for v4

1. Launch / continue v4 training on A800-D GPU 0–3/6.
2. Evaluate the first v4 checkpoint on the same S2/Seq1 protocol used above.
3. Compare clean MPJPE against **9.03 mm** and 2-view MPJPE against the v2/v3 curves in this table.
4. Update this doc once a converged v3 or v4 checkpoint becomes available.
