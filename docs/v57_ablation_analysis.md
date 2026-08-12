# v57 Domain-Conditional Physical-Space Calibration — Ablation Analysis

**Date:** 2026-08-11
**Status:** Living document — updated as ablations finish.
**Protocol:** H36M true-GT standard (S1, S5, S6, S7, S8 train → S9, S11 test)
**Manifest:** `configs/splits/h36m_true_gt_standard.yaml`
**Main run:** `scripts/run_v57_h36m_true_gt_medium.sh`

This document explains what each v57-related ablation is testing and how to interpret its results, so future agents can decide which variant to promote, merge, or abandon.

---

## 1. What v57 is

v57 = **Domain-Conditional Physical-Space Calibration (DC-PSC)**.
It is the newest head in the `OmniMultiViewFusionV5` family and sits on top of the v25 geometry-fusion + v45 adaptive fusion + v46 sparse-view generalisation + v50 self-evolution feedback + v51 cross-domain sparse-view reliability + v52 uncertainty-weighted triangulation stack.

The DC-PSC head adds a small domain-conditional MLP that predicts a physical-space residual correction after triangulation, plus auxiliary losses:

* floor-plane penalty,
* bone-scale consistency,
* 2-D reprojection consistency of the corrected pose.

Key flags: `--use_v57_domain_conditional_psc`, `v57_dcpsc_loss_weight`, `v57_dcpsc_floor_weight`, `v57_dcpsc_bone_weight`, `v57_dcpsc_reproj_weight`.

The medium H36M true-GT run is done and achieved:

| Method | Best val MPJPE (mm) | Best epoch | Final epoch | Status |
|---|---:|---:|---:|---|
| v25 | 72.80 | 2 | 207.62 | Diverges |
| v80 | 39.98 | 4 | 133.71 | Overfits after epoch 4 |
| **v57** | **75.16** (obs.) / **81.47** (ckpt) | **3** | **80.21** | **Early stopped at epoch 5** |

*Note:* The observed best val MPJPE is 75.16 mm at epoch 3, but the saved checkpoint corresponds to **epoch 2 (81.47 mm)** because the epoch-3 val-loss improvement (0.000384) was below `early_stopping_min_delta=0.001`. See `docs/v57_checkpoint_validation.md`.

See `docs/v57_v80_v25_comparison.md` for the full comparison and `docs/results_true_gt_h36m.md` for the leaderboard.

---

## 2. Why we are ablating v57

The medium run shows v57 is **not yet competitive** with v80 (75.16 mm vs. 39.98 mm) and only marginally better than the diverged v25. The ablations answer four questions:

1. **Is the DC-PSC head itself harmful, or is it the surrounding hyper-parameter / capacity mismatch?**
2. **Do the floor/bone auxiliary losses help, or do they destabilise training?**
3. **Does the v57 correction need to be bounded / clamped?**
4. **Does adding lightweight temporal aggregation improve v57?**

---

## 3. Ablation matrix

Each row is a named run script in `scripts/`. All are CPU/GPU-smoke or small-scale runs intended to isolate a single design choice.

| # | Script | Name | What it tests | Key change vs. v57 medium |
|---|--------|------|--------------|---------------------------|
| 1 | `run_v57_h36m_true_gt_medium.sh` | v57 H36M true-GT medium | Baseline v57 on true-GT H36M | Full v57, d=128, train_samples=1024 |
| 2 | `run_v57_domain_conditional_psc_smoke_local_4090.sh` | v57 smoke (WebBridge mixed) | Whether v57 can train at all on the WebBridge mixed loader | d=64, smoke, H36M+MPI mixed |
| 3 | `run_v57_domain_conditional_psc_medium_local_4090.sh` | v57 medium (WebBridge mixed) | Whether a smaller v57 on mixed data beats the H36M-only medium run | d=64, 3 epochs, 50 samples/epoch |
| 4 | `run_v57_noncircular_smoke_local_4090.sh` | v57 non-circular smoke | Training on non-circular labels only (MPI + Shelf/Campus, no H36M) | Different manifest, small smoke |
| 5 | `run_v57_mpi_only_noncircular_smoke_local_4090.sh` | v57 MPI non-circular smoke | MPI-only non-circular data | MPI-only manifest |
| 6 | `run_v57_mpi_only_noncircular_medium_local_4090.sh` | v57 MPI non-circular medium | MPI-only medium run | MPI-only, medium scale |
| 7 | `run_v61_dcpsc_sefh_uwt_feedback_tiny_local_4090.sh` | v61 tiny | v57 + v60 SEFH→UWT feedback head | Adds `use_v60_sefh_uwt_feedback` |
| 8 | `run_v62_dcpsc_sefh_uwt_feedback_gradstop_tiny_local_4090.sh` | v62 grad-stop | Stop v57 PSC gradients to base network | `--v57_dcpsc_stop_grad_to_base` |
| 9 | `run_v63_dcpsc_sefh_uwt_feedback_clamp_tiny_local_4090.sh` | v63 clamp | Clamp v57 correction + grad-stop | `--v57_dcpsc_max_correction 0.5` + grad-stop |
| 10 | `run_v66_dcpsc_residual_only_v60_tiny_local_4090.sh` | v66 residual-only | Remove floor/bone heads, keep residual correction | `--no_v57_dcpsc_use_floor --no_v57_dcpsc_use_bone_scale` + grad-stop |
| 11 | `run_v67_dcpsc_identity_clamp_v60_tiny_local_4090.sh` | v67 identity-clamp | Disable PSC auxiliary loss, keep tiny residual correction | All PSC loss weights = 0.0, max_correction=0.5 |
| 12 | `run_v78_v57_dcpsc_200samples_3epochs_local_4090.sh` | v78 200-sample | Slightly larger than smoke, short run | 200 samples, 3 epochs |
| 13 | `run_v74_v57_v49_lite_temporal_smoke_local_4090.sh` | v74 + v49 temporal | Add v49-Lite causal temporal aggregation to v57 | `use_v49_lite_temporal_aggregation` |

---

## 4. What each ablation is testing (scientific rationale)

### 4.1 v57 on different data splits

* **`run_v57_h36m_true_gt_medium.sh`** — the canonical test. If v57 cannot beat v80 here, the architecture is not yet useful for the paper's main benchmark.
* **`run_v57_domain_conditional_psc_smoke_local_4090.sh`** / `medium_local_4090` — tests whether the WebBridge mixed loader (H36M + MPI) gives v57 enough diversity to learn the physical-space prior. If this smoke converges while the H36M-only run does not, the problem is **data quantity/diversity**, not the head.
* **`run_v57_noncircular_smoke_local_4090.sh`** — tests training without any H36M circular labels. Useful because the old H36M labels were circular (DLT of 2D). If v57 works here, it proves the head does not depend on circular supervision.
* **`run_v57_mpi_only_noncircular_*.sh`** — tests whether MPI-INF-3DHP alone, with real mocap labels, is sufficient to train v57.

### 4.2 v57 loss and gradient ablations

* **`run_v61_dcpsc_sefh_uwt_feedback_tiny_local_4090.sh`** — adds a v60 feedback path from the Self-Evolution Feedback Head (SEFH) into the Uncertainty-Weighted Triangulation (UWT) block. Tests whether richer feedback before the PSC head improves stability.
* **`run_v62_dcpsc_sefh_uwt_feedback_gradstop_tiny_local_4090.sh`** — same as v61 but stops gradients from the PSC loss to the base network. Tests whether the PSC loss destabilises the shared base by back-propagating too aggressively.
* **`run_v63_dcpsc_sefh_uwt_feedback_clamp_tiny_local_4090.sh`** — clamps the v57 physical-space correction to ±0.5 m and stops gradients. Tests whether the correction explodes without an explicit bound.
* **`run_v66_dcpsc_residual_only_v60_tiny_local_4090.sh`** — removes floor and bone-scale auxiliary losses, keeping only the gated residual correction. Tests whether the auxiliary losses are necessary or harmful.
* **`run_v67_dcpsc_identity_clamp_v60_tiny_local_4090.sh`** — keeps the residual head but disables all PSC loss terms (loss_weight = 0.0). Tests whether a tiny, identity-initialised, clamped correction can learn anything useful from the main MSE alone.

### 4.3 Architecture and temporal ablations

* **`run_v74_v57_v49_lite_temporal_smoke_local_4090.sh`** — adds a lightweight causal temporal smoothing module on top of v57. Tests whether temporal consistency helps without hurting per-frame MPJPE.

---

## 5. How to interpret the results

### 5.1 Primary metric

Best **combined direct MPJPE (mm)** on the H36M true-GT val split (S9 + S11). Lower is better.

### 5.2 Stability metric

Compare **final-epoch MPJPE** to **best-epoch MPJPE**.

| Condition | Interpretation |
|-----------|---------------|
| final ≤ best × 1.10 | Stable; the model is not overfitting catastrophically. |
| best < final ≤ best × 1.25 | Mild overfit; may still be usable if best epoch is saved. |
| final > best × 1.25 or NaN/Inf | Unstable; the run is diverging. |

### 5.3 Decision gates

After each ablation, use these gates:

1. **If v57 H36M true-GT medium drops below 55 mm and stays stable**, the base recipe is already good enough and only hyper-parameter tuning is needed.
2. **If the WebBridge mixed smoke/medium run is substantially better than the H36M-only run**, the problem is data diversity. Promote mixed-dataset training for v57.
3. **If v62 (grad-stop) is more stable than v61 (full PSC loss)**, the PSC loss is back-propagating harmful gradients to the base. Make `--v57_dcpsc_stop_grad_to_base` the default.
4. **If v63 (clamp + grad-stop) is better than v62**, the correction magnitude is exploding. Make `--v57_dcpsc_max_correction` the default.
5. **If v66 (residual-only) matches or beats v63**, the floor/bone auxiliary losses are unnecessary or harmful. Simplify the head.
6. **If v67 (identity-clamp, no PSC loss) is comparable to v63/v66**, the main MSE is enough to train a small correction. Consider dropping the PSC auxiliary losses entirely.
7. **If v74 (temporal) improves over the matching non-temporal run**, add v49 temporal aggregation to the v57 recipe.

### 5.4 Relative targets

| Target | MPJPE | Meaning |
|--------|------:|---------|
| Leader (Iskakov) | ~23.35 mm | Long-term goal; v57 may not reach this. |
| Confidence-weighted DLT | ~25.87 mm | Strong geometric baseline. |
| v80 medium | 39.98 mm | Immediate peer target for v57. |
| v25 medium | 72.80 mm | Lower bound; v57 should beat this. |
| v57 medium | 75.16 mm | Current baseline for v57. |

A v57 ablation is considered **promising** if it lands within 10–15 mm of v80 (i.e., ≤ 55 mm). Ablations above 72 mm are not improving over the un-regularised v25/v57 starting point.

---

## 6. Recording results

When an ablation finishes, append a row to the table below and update `docs/results_true_gt_h36m.md` if the run used the H36M true-GT val split.

| Run | Best MPJPE | Best epoch | Final MPJPE | S9 direct | S11 direct | Stable? | Notes |
|-----|-----------:|-----------:|------------:|----------:|-----------:|---------|-------|
| v57 H36M true-GT medium | 75.16 (obs.) / 81.47 (ckpt) | 3 | 80.21 | — | — | Yes | Early stopped at epoch 5; saved ckpt is epoch 2 |

For non-H36M runs (MPI, WebBridge mixed), record the val MPJPE from the relevant manifest and note the dataset.

---

## 7. Safety / constraints

* The local RTX 4090 can run **only one training task at a time**.
* Do not start any of these ablations while another agent's GPU run is active.
* A800-D and the Docker `motionflow` service are **read-only**; do not write or launch anything there.
* CPU-only preparation (writing scripts, log parsing) is allowed while the GPU is busy.

---

## 8. Related files

* `docs/v57_v80_v25_comparison.md` — detailed v57 vs. v80 vs. v25 analysis.
* `docs/results_true_gt_h36m.md` — main true-GT leaderboard.
* `configs/benchmark_v57_h36m_true_gt_medium.yaml` — reference config for the medium run.
* `scripts/run_v57_h36m_true_gt_medium.sh` — main v57 medium launch script.
