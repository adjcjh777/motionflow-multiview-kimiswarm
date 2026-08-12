# v57 vs. v80 vs. v25 H36M True-GT Comparison

**Date:** 2026-08-11
**Protocol:** H36M true-GT standard (S1, S5, S6, S7, S8 train → S9, S11 test)
**Manifest:** `configs/splits/h36m_true_gt_standard.yaml`

## 1. Results at a Glance

| Method | Best val MPJPE (mm) | Best epoch | Final epoch | Model params | Status |
|---|---:|---:|---:|---:|---|
| **v80** | **39.98** | 4 | 133.71 (epoch 8) | 817,919 | Diverges after epoch 4 |
| **v25** | **72.80** | 2 | 207.62 (epoch 8) | 2,731,695 | Diverges after epoch 2 |
| **v57** | **75.16** (obs.) / **81.47** (ckpt) | 3 | 80.21 (epoch 5) | 3,728,222 | Early stopped; plateau / mild rise; saved ckpt is epoch 2 |

For reference:

- Iskakov ICCV 2019: **23.35 mm**
- Confidence-weighted DLT: **25.87 mm**
- Unweighted DLT: **29.19 mm**

*Checkpoint note:* v57's observed best val MPJPE is **75.16 mm** at epoch 3, but the saved checkpoint corresponds to **epoch 2 (81.47 mm)** because the val-loss improvement was below `early_stopping_min_delta=0.001`. See `docs/v57_checkpoint_validation.md`.

All three MotionFlow variants lag behind the geometric / learnable-triangulation baselines, and all begin degrading after a small number of epochs. The central puzzle is why **v57**, the newest architecture, ended up slightly worse than the much smaller **v80** and only marginally better than the un-regularised **v25**.

## 2. What Each Variant Actually Is

### v25 — multiview geometry fusion (baseline)

- Adds `MultiviewGeometryFusionV25` on top of `OmniMultiViewFusionV5`.
- Features: geometry-aware cross-view attention, learned depth triangulation, geometry bundle adjustment.
- Config in this run: `d=128`, `residual_hidden=256`, `n_st_layers=3`, `use_v45_adaptive_geometry_fusion=false`.
- v25-only loss weight: `v25_geom_loss_weight=0.1`.

### v80 — view-reliability branch (best known)

- Same v25 geometry fusion **plus** `v80_view_reliability` (per-view / per-joint reliability weighting).
- Smaller backbone: `d=64`, `residual_hidden=128`, `n_st_layers=2`.
- v80 loss weight: `v80_vrbt_loss_weight` defaults; the module produces reliability-weighted triangulation.

### v57 — domain-conditional physical-space calibration (DC-PSC)

- Builds on the v25+v45+v46+v50+v51+v52 feature stack **plus** a domain-conditional physical-space calibration head (`use_v57_domain_conditional_psc=true`).
- Larger backbone: `d=128`, `residual_hidden=256`, `n_st_layers=3` (same as v25).
- Extra loss terms: floor, bone-scale, reprojection, with `v57_dcpsc_loss_weight=0.1` and a 1-epoch warmup.
- Also enables robust DLT / IRLS reweighting and full-precision DLT, which v80 in this run does **not** use.

## 3. Why v57 Underperformed

### 3.1 Capacity–data mismatch is larger for v57

v57 has **3.73 M parameters** (3.7× more than v80, 1.4× more than v25) and the same `train_samples=1024` per epoch as the others. With only 64 gradient steps per epoch, a larger model simply memorises the tiny training slice faster; the extra physical-space head cannot overcome the limited data.

### 3.2 The v57 physical-space head is under-regularised and under-warmed

Compared with the v53 physical-space calibration block (loss weight 1.0, warmup 0 epochs in the disabled code path), v57 uses:

```json
"v57_dcpsc_loss_weight": 0.1,
"v57_dcpsc_warmup_epochs": 1,
```

While the warmup is sensible, the loss weight of `0.1` means the physical-space regularisation only exerts a mild force. More importantly, the DC-PSC head is trained jointly with the rest of the network from the start (only the loss is warmed up). A 3.7 M-parameter model therefore has many additional degrees of freedom that can drift on a small dataset before the calibration loss has a strong effect.

### 3.3 Robust DLT + IRLS reweighting adds noise in the small-sample regime

The v57 run is the only one of the three with:

```json
"use_full_precision_dlt": true,
"use_robust_dlt_reweight": true,
"use_irls_reweight": true
```

Robust triangulation is valuable with real, noisy detections, but on the small true-GT medium split it can discard too many training samples or assign unstable weights. This instability compounds with the extra v57 head and helps explain why v57 starts worse than v25/v80 at epoch 1 (98.11 mm vs. 83.19 mm / 88.78 mm).

### 3.4 v57 converges more slowly and early-stops before the best v80-like region

| Epoch | v57 MPJPE | Δ vs. previous |
|---:|---:|---:|
| 1 | 98.11 | — |
| 2 | 81.47 | −16.64 |
| 3 | **75.16** | −6.31 |
| 4 | 76.60 | +1.44 |
| 5 | 80.21 | +3.61 |

v57 is still improving through epoch 3, but the improvement is small. Its early-stopping patience of 3 epochs triggers at epoch 5 because of the mild rise. In contrast, v80 gets to **39.98 mm** at epoch 4 because it starts from a much better place (66.26 mm at epoch 2) and has less capacity to overfit.

### 3.5 The v80 run already contains the key regularisation that v57 lacks

The v80 run in `outputs/omniview_fusion_v80_h36m_true_gt_medium.log` uses the view-reliability branch (`use_v80_view_reliability=true`), which explicitly down-weights unreliable views. This is exactly the kind of regularisation that benefits a true-GT protocol where some camera views are noisier than others. v57 has no equivalent explicit per-view reliability mechanism (the v52 uncertainty-weighted triangulation is present in both v57 and v80, but only v80 also has the v80 reliability branch in this comparison).

### 3.6 Hyperparameters are tuned for v80, not v57

The medium run script was originally written for v80. v57 inherited the same hyperparameters:

```json
"lr": 0.001,
"train_samples": 1024,
"weight_decay": 0.0,
"early_stopping_patience": 3,
"outlier_view_prob": 0.3
```

These are the same settings that cause v25 to diverge. v80 survives them because it is smaller; v57, despite its physical-space regularisation, does not.

## 4. Architecture Differences Summary

| Aspect | v25 | v80 | v57 |
|---|---|---|---|
| Backbone width `d` | 128 | 64 | 128 |
| Backbone depth `n_st_layers` | 3 | 2 | 3 |
| Residual hidden | 256 | 128 | 256 |
| Total params | 2.73 M | 0.82 M | 3.73 M |
| v25 geometry fusion | yes | yes | yes |
| v80 view reliability | no | yes | no |
| v57 DC-PSC | no | no | yes |
| v45 adaptive fusion | no | yes | yes |
| v46 sparse-view gen | no | yes | yes |
| v50 self-evolution | no | yes | yes |
| v51 cross-domain reliability | no | yes | yes |
| v52 uncertainty triangulation | no | yes | yes |
| v30 hierarchical multiview | no | no | no |
| Robust DLT / IRLS | yes | no | yes |
| Weight decay | 0.0 | 0.0 | 0.0 |
| Train samples / epoch | 1024 | 1024 | 1024 |

The extra v57 components (v50/v51/v52/v52 + DC-PSC) add regularisation *in principle*, but the **net capacity increase** outweighs the regularisation benefit on this small dataset.

## 5. Conclusions

1. **v57 is not inherently worse; it is over-parameterised for the medium split.** With only 1024 samples/epoch, a 3.7 M-parameter model overfits before the physical-space calibration head can stabilise it.

2. **v80 wins because it is smaller and has the view-reliability regulariser.** A smaller model + explicit per-view reliability weighting is a better match for limited true-GT data.

3. **v25 and v57 share the same core problem:** too many parameters, too little regularisation, too few samples per epoch, and no early stop at the right moment. v57’s extra physical-space losses are not strong enough to fix this.

4. **The true-GT protocol is unforgiving.** Iskakov and DLT are strong because they directly exploit multi-view geometry with almost no learnable parameters. Learned MotionFlow variants must be heavily regularised to compete.

## 6. Recommended Next Steps

To make v57 competitive, apply the same fixes that are already planned for v25:

1. **Increase samples per epoch** to at least `4096` (matching the v25 re-run proposal in `docs/v25_divergence_diagnosis.md`).
2. **Add weight decay** (`1e-4`) and a lower learning rate (`5e-4`).
3. **Reduce outlier augmentation** on the small split (`outlier_view_prob 0.15`).
4. **Increase the v57 DC-PSC loss weight** or keep it warm for longer so the physical-space head can actually regularise the network.
5. **Consider progressive unfreezing:** train the v57 calibration head for 1–2 epochs before unfreezing the base network.
6. **Run a smoke ablation** that disables v57 DC-PSC but keeps the rest of the v57 stack, to isolate whether the head itself is harmful or only the capacity mismatch is.

## 7. Files Referenced

- `outputs/omniview_fusion_v25_h36m_true_gt_medium.{log,config.json}`
- `outputs/omniview_fusion_v80_h36m_true_gt_medium.{log,config.json}`
- `outputs/omniview_fusion_v57_h36m_true_gt_medium.{log,config.json}`
- `docs/results_true_gt_h36m.md`
- `docs/v25_divergence_diagnosis.md`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
