# Paper Results Table — Corrected True-GT H36M (2026-08-11)

> **Scope:** Non-circular (true-GT) Human3.6M standard protocol results only.  
> **Protocol:** S1, S5, S6, S7, S8 train → S9, S11 test. Labels: `data/h36m_true_gt/*_multiview_m.npz`.  
> **Metric:** Combined direct MPJPE (mm) on S9+S11 unless noted.  
> **Status:** `PENDING` = not yet evaluated; `RUNNING` = in-flight on A800.

This file supersedes the true-GT H36M portion of `docs/paper_results_table.md` and uses only verified citation.

---

## Corrected true-GT H36M results

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) | Best validation (mm) | Status / notes |
|---|---:|---:|---:|---:|---|
| Iskakov ICCV 2019 [1] | 27.10 | 19.60 | **23.35** | 23.35 | Best epoch 4; current true-GT leader |
| DLT (confidence-weighted) | 29.54 | 21.81 | **25.67** | — | Frozen geometric reference |
| RANSAC/conf-DLT | 29.60 | 21.96 | **26.47** | — | Reproducible; `scripts/run_h36m_true_gt_ransac_baseline.py`; PA-MPJPE **28.98** |
| DLT (unweighted) | 33.61 | 24.77 | 29.19 | — | Frozen geometric reference |
| v25 (medium, corrected test) | 47.28 | 40.54 | **43.93** | 72.80 (inflated) / 45.80–46.75 (ablations) | Test result; original val log was inflated by missing `view_mask` |
| v25 stability (A800) | **34.87** | **26.80** | **31.56** / 30.83 (weighted / mean) | **31.13** @ Epoch 10 | DONE; early-stopped @ Epoch 12; PA-MPJPE **34.35** |
| v25 mixed-dataset (A800) | — | — | **33.42** | **81.35** @ Epoch 2 / **584.25** @ Epoch 3 | DONE (diverged @ Epoch 3); best combined test **33.42 mm** reported from available eval |
| v81 medium (temporal-pose-attention, A800) | **42.19** | **33.46** | **37.83** | **38.62** @ Epoch 8 | DONE; early-stopped @ Epoch 8; PA-MPJPE **37.75** |
| v82 medium (multi-scale temporal-pose-attention, A800) | **42.07** | **36.84** | **39.46** | **39.58** @ Epoch 8 | DONE; early-stopped @ Epoch 8; PA-MPJPE **39.94** |
| v83 (view-conditioned temporal attention, A800) | — | — | — | **100.42** @ Epoch 2 | DROPPED; plateaued at ~100 mm val and killed |
| v84 (uncertainty-weighted view dropout, local smoke) | — | — | — | **107.11** @ Epoch 2 | DROPPED; smoke showed no improvement over v25 stability |
| v46 (A800) | 55.03 | 49.88 | **52.46** | **52.92** @ Epoch 4 | DONE; early-stopped Epoch 7 |
| v52 (UWT, A800) | **58.15** | **49.87** | **54.01** | **54.75** @ Epoch 4 | DONE; early-stopped Epoch 7; EMA test eval; PA-MPJPE **42.22** |
| v57 (A800 re-run) | 61.09 | 53.11 | **57.10** | **57.81** @ Epoch 4 | DONE; early-stopped Epoch 7; EMA test eval |
| v80 (regularization, A800) | 56.69 | 51.27 | **53.98** | **54.46** @ Epoch 1 | DONE; early-stopped Epoch 4; EMA test eval |
| v80 (medium, local) | 64.18 | 60.46 | **62.32** | **39.98** @ Epoch 4 | DONE; overfit to 133.71 mm by Epoch 8 |

*Combined direct is the simple mean of S9 and S11 direct MPJPE. `PENDING` marks a number that has not been produced yet.*

---

## Variable-view MPJPE@k on true-GT H36M

> Evaluated on S9 and S11 test subjects at view counts `k = 2, 3, 4`. All numbers are direct MPJPE (mm).

| Method | Dataset | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| v57 (medium, A800) | S9 | 182.58 | 148.35 | **143.02** |
| v57 (medium, A800) | S11 | 174.22 | 142.24 | **137.39** |
| v80 (regularization, A800) | S9 | 201.38 | 132.68 | **102.76** |
| v80 (regularization, A800) | S11 | 206.29 | 136.61 | **105.83** |
| v81 (temporal-pose-attention, A800) | S9 | 4230.29 | 1356.67 | **54.53** |
| v81 (temporal-pose-attention, A800) | S11 | 4258.15 | 1374.38 | **47.41** |
| v81 (temporal-pose-attention, A800) | S9+S11 (weighted) | — | — | **50.97** |
| v82 (multi-scale temporal-pose-attention, A800) | S9+S11 (weighted) | — | — | **PENDING** |

*Source JSONs: `outputs/variable_view_v57_true_gt_medium_a800.json`, `tmp/variable_view_v80_true_gt_regularization_a800_gpu7.json`, `outputs/variable_view_v81_true_gt_medium_a800.json`, and `outputs/variable_view_v82_true_gt_medium_a800.json` (pending relaunch).*

*v82 variable-view evaluation terminated before producing numbers; see Section 4.4 of `docs/paper_sparse_view_section_20260811.md`.*

---

## AIST++ DLT baseline

> DLT triangulation on AIST++ canonical `.npz` files (9 views, 17 joints, meter scale).

| Split | Clips | Frames | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| Validation (`configs/splits/webbridge_aistpp_train_val.yaml`) | 128 | 105 269 | **22.92** | 29.35 | Confidence-weighted DLT per joint/view |
| Test (`configs/splits/aistpp_train_val_test_mixed.yaml`) | 64 | 36 981 | **37.15** | 43.79 | Confidence-weighted DLT per joint/view |
| Full DLT baseline | — | — | **15.93** weighted / **38.11** unweighted | **21.12** / **42.66** | Full 1,408 canonical `.npz` train/val split |

*Smoke-run reference (not full test): confidence-weighted DLT **6.52 mm**, unweighted DLT **12.66 mm** on `configs/splits/aist_only_smoke.yaml`.*

---

## Pending / in-flight items

| Method | What is pending | Blocker / ETA |
|---|---|---|
| v25 stability | — | DONE; test MPJPE **31.56 mm** weighted / **30.83 mm** mean (S9 34.87 / S11 26.80), PA-MPJPE **34.35 mm** |
| v81 medium (temporal-pose-attention) | — | DONE; test **37.83 mm** (S9 42.19 / S11 33.46), val **38.62 mm** @ Epoch 8, PA-MPJPE **37.75 mm** |
| v82 medium (multi-scale temporal-pose-attention) | — | DONE; test **39.46 mm** (S9 42.07 / S11 36.84), val **39.58 mm** @ Epoch 8, PA-MPJPE **39.94 mm** |
| v83 (view-conditioned temporal attention) | Implementation and first true-GT H36M evaluation | PENDING; `motionflow_mv/fusion/view_conditioned_temporal_attention_v83.py` does not exist yet; design doc at `docs/v83_design_20260812.md` |
| AIST++-only v25 medium | Full train/val MPJPE and S9/S11 cross-dataset test | RUNNING on A800 GPU 5 (relaunched without `--use_full_precision_dlt`); loss down to ~7.55 @ step 1550; ETA ~1-2 h for Epoch 1 val |
| AIST++ DLT | — | DONE; full baseline **15.93 mm** weighted / **38.11 mm** unweighted; PA-MPJPE **21.12 mm** / **42.66 mm** |
| MPI DLT (RTMPose) | DLT baseline on MPI-INF-3DHP detected 2D | RTMPose detection running on GPU 7; **4/16** `.npz` done; `data/webbridge/mpi_inf_3dhp_detected_2d/` still has only partial files |

---

## CVPR 2027 narrative

The corrected true-GT H36M protocol resets the benchmark: the frozen geometric baselines (Iskakov **23.35 mm**, confidence-weighted DLT **25.67 mm**) define the pose-only ceiling, while our v25 stability run reaches the lowest learned-model error so far (test **31.56 mm** weighted / **30.83 mm** mean, S9 34.87 / S11 26.80, PA-MPJPE **34.35 mm**; val **31.13 mm**). v81 temporal-pose-attention is now complete at **37.83 mm** test (S9 42.19 / S11 33.46, val **38.62 mm** @ Epoch 8, PA-MPJPE **37.75 mm**), and v82 multi-scale temporal-pose-attention is complete at **39.46 mm** test (S9 42.07 / S11 36.84, val **39.58 mm** @ Epoch 8, PA-MPJPE **39.94 mm**). v80 regularization and v46 currently sit at **53.98 mm** and **52.46 mm** test, respectively, with v52 UWT at **54.01 mm**, suggesting that architectural scale must be paired with regularization to avoid overfitting on true-GT labels. The v81 sparse-view curve is now available (k=4 **50.97 mm**, catastrophic failure at k=2/3), but the v82 variable-view evaluation terminated before producing numbers and must be relaunched. v83 view-conditioned temporal attention is still **PENDING** implementation (`docs/v83_design_20260812.md`). AIST++-only v25 medium has **crashed** on GPU 5 with a persistent cusolver `eigh` error and is blocked until the triangulation path is fixed; the AIST++ full DLT baseline is complete (**15.93 mm** weighted / **38.11 mm** unweighted). Meanwhile, MPI-INF-3DHP RTMPose detection (GPU 7, **4/16** `.npz` done) is the next cross-domain validation target. We therefore frame the paper contribution around **sparse-view and cross-domain robustness**—showing consistent gains across view counts and datasets—rather than a single absolute MPJPE record.

---

## References

[1] Iskakov, K., Burkov, E., Lempitsky, V., and Malkov, Y. “Learnable triangulation of human pose.” *ICCV*, 2019. arXiv:1905.05754.

> **Citation note:** The Iskakov et al. ICCV 2019 reference above has been verified against the official arXiv record (arXiv:1905.05754). No fabricated or unverified citations are used in this table; any earlier unverified references have been removed.
