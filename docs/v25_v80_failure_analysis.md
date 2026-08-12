# v25 vs. v80 H36M True-GT Failure Analysis

**Date:** 2026-08-11  
**Scope:** Explain why both `v25` (multi-view geometry fusion) and `v80` (view-reliability before triangulation) overfit on the repaired H36M true-GT standard protocol, and why `v80` reaches a much lower best validation MPJPE (`39.98 mm`) than `v25` (`72.80 mm`) under the same medium-run budget.  
**Key logs:**

- `outputs/omniview_fusion_v25_h36m_true_gt_medium.log`
- `outputs/omniview_fusion_v80_h36m_true_gt_medium.log`
- Saved configs: `outputs/omniview_fusion_v25_h36m_true_gt_medium.config.json`, `outputs/omniview_fusion_v80_h36m_true_gt_medium.config.json`

---

## 1. Observed behaviour

Both medium runs were executed on the local RTX 4090 with the same data split (`configs/splits/h36m_true_gt_standard.yaml`), the same number of training samples per epoch (`--train_samples 1024`), the same batch size (`16`), the same learning rate schedule (`lr=1e-3`, cosine, 1-epoch warmup), and the same view/augmentation settings.  The only intentional differences are the model architecture flags.

| Run | Params | Best val MPJPE | Best epoch | Final val MPJPE | Trajectory |
|-----|-------:|---------------:|-----------:|----------------:|------------|
| v25 | 2,731,695 | **72.80 mm** | 2 | 207.62 mm (epoch 8) | improves epoch 1→2, then monotone explosion |
| v80 |   817,919 | **39.98 mm** | 4 | 133.71 mm (epoch 8) | improves through epoch 4, then slower rise |

The raw epoch curves are:

**v25**

| Epoch | train_loss | val_loss | val_MPJPE (mm) |
|------:|-----------:|---------:|---------------:|
| 1 | 6.462 | 0.002519 | 83.19 |
| 2 | 6.475 | 0.001976 | **72.80** |
| 3 | 6.148 | 0.002067 | 73.12 |
| 4 | 5.930 | 0.002451 | 78.33 |
| 5 | 5.861 | 0.003625 | 88.38 |
| 6 | 5.855 | 0.007086 | 115.98 |
| 7 | 5.743 | 0.013444 | 159.43 |
| 8 | 5.782 | 0.021829 | 207.62 |

**v80**

| Epoch | train_loss | val_loss | val_MPJPE (mm) |
|------:|-----------:|---------:|---------------:|
| 1 | 6.426 | 0.002875 | 88.78 |
| 2 | 6.784 | 0.001722 | 66.26 |
| 3 | 6.013 | 0.000933 | 44.41 |
| 4 | 5.517 | 0.000851 | **39.98** |
| 5 | 5.299 | 0.001649 | 56.68 |
| 6 | 5.188 | 0.003270 | 83.30 |
| 7 | 5.165 | 0.005446 | 110.36 |
| 8 | 5.161 | 0.007782 | 133.71 |

Both runs show **falling training loss** alongside a **valley-shaped validation MPJPE** followed by overfitting.  The crucial distinction is that `v80`’s valley is deeper (`39.98 mm` vs `72.80 mm`) and occurs two epochs later, after which it degrades more slowly.

---

## 2. Why both models overfit on true GT

### 2.1 The true-GT labels removed the circular-label safety net

Under the old circular-label H36M setup, `joints_3d == DLT(points_2d, cameras)`.  Any model that simply learned to reproduce the DLT triangulation could report near-zero MPJPE.  With the repaired true-GT labels (`data/h36m_true_gt/*_multiview_m.npz`), the labels are real mocap world coordinates.  The model can no longer cheat by memorising the DLT layer; it must learn a genuine mapping from multi-view 2D keypoints to 3D pose.  This exposes every form of overfitting that the circular labels had masked.

### 2.2 Far too few unique training samples per epoch

Both scripts use:

```bash
--train_samples 1024 --batch_size 16 --epochs 8
```

That is **64 gradient steps per epoch** on a model with hundreds of thousands to millions of parameters.  The full training set contains ~390 k frames, so each epoch samples only **~0.26 %** of the available data.  With so few unique examples, the model quickly memorises the small epoch-specific distribution rather than learning a stable estimator.

### 2.3 No weight decay and no early stopping

Both saved configs contain:

```json
"weight_decay": 0.0,
"early_stopping_patience": 0,
"early_stopping_min_delta": 0.0
```

- **No L2 regularisation** leaves all residual MLPs, attention layers, and geometry heads free to over-fit.
- **No early stopping** forces the trainer to continue for the full 8 epochs, well past the best validation point.  The best checkpoints are preserved only because the trainer separately saves the lowest `val_MPJPE` model.

### 2.4 Aggressive learning rate for the data size

Both runs use `lr=1e-3` with a 1-epoch cosine warmup.  After the warmup the cosine schedule is already at a relatively high value, so the model takes large steps in parameter space while it has seen only 1–2 k unique examples.  This magnifies any overfitting signal.

### 2.5 Strong augmentation relative to dataset size

Both runs enable:

```bash
--outlier_view_prob 0.3 --outlier_view_max_views 1 \
--outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
--use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4
```

Injecting corrupted views in 30 % of samples and randomly subsampling to 2–4 views is a useful regulariser when data is abundant, but with only 1,024 training samples per epoch the augmentation becomes a large fraction of the signal.  The model learns to fit the augmentation pattern rather than the underlying geometry.

### 2.6 Identity-at-init modules still overfit once their gates open

Both architectures contain identity-initialized gates (e.g. `residual_scale` in `DepthProposalTriangulation`, zero-initialized final layers in `v80`).  These gates start by preserving the DLT baseline, which is why both models begin near the DLT/Iskakov ballpark.  Once training proceeds, the gates open and the learned components start to specialise to the tiny training set, causing validation MPJPE to rise.

---

## 3. Why v80 (39.98 mm) outperforms v25 (72.80 mm)

The two runs are not just two independent models; `v80` is essentially `v25` *plus* a stack of regularising modules.  The v80 script enables:

```bash
--use_multiview_geometry_fusion_v25 \
--use_v45_adaptive_geometry_fusion \
--use_v46_sparse_view_generalization \
--use_v50_self_evolution_feedback_head \
--use_v51_cross_domain_sparse_view_reliability \
--use_v52_uncertainty_weighted_triangulation \
--use_v80_view_reliability
```

whereas the v25 script enables only the v25 geometry-fusion block.  This architectural difference explains the performance gap.

### 3.1 Lower base capacity

| Config | v25 | v80 |
|--------|----:|----:|
| `d` | 128 | 64 |
| `residual_hidden` | 256 | 128 |
| `n_st_layers` | 3 | 2 |
| Total params | 2,731,695 | 817,919 |

`v25` uses a larger base feature backbone (`d=128`, `residual_hidden=256`, 3 ST layers).  With only 64 gradient steps per epoch, this higher capacity is pure liability: the v25 geometry head has enough degrees of freedom to fit the small training set almost exactly, then generalise poorly.  `v80` uses a much smaller backbone and delegates refinement to small, specialised heads.

### 3.2 Multiple identity-initialised regularisation heads

`v80` stacks several lightweight, identity-initialised modules:

- **v45 adaptive geometry fusion** (`AdaptiveGeometryFusionV45`) predicts per-view / per-joint reliability weights from reprojection residuals and re-triangulates.
- **v46 sparse-view generalisation** randomly drops views during training (view dropout `p=0.3`, min 2 views) and learns to handle variable view counts.
- **v52 uncertainty-weighted triangulation** predicts per-view precision and performs a weighted, damped triangulation, with an small regularising loss (`v52_uwt_loss_weight=0.01`).
- **v80 view-reliability head** predicts a pre-triangulation reliability prior for v52, also with identity-initialised gates and a clamped minimum weight.
- **v50/v51 auxiliary heads** are also instantiated (training-only; their loss weights are set to `0.0` in the medium script, so they do not drive gradients but belong to the same sparse-view robustness stack).

All of these modules start as near-identity and can only deviate where the data supports it.  They act as an **ensemble of regularisers** around the DLT baseline, whereas v25 has only the single v25 geometry head with no comparable regularisation stack.

### 3.3 The v80 loss landscape is better conditioned

The v80 pipeline explicitly reasons about **per-view reliability weights** before and during triangulation.  This is a far more constrained problem than v25’s free-form geometry refinement:

- v25’s `DepthProposalTriangulation` directly proposes new 3-D points along learned depth hypotheses and adds an unbounded residual (`pred_3d + residual_scale * residual`).
- v80’s heads predict scalar weights and leave the actual triangulation to a weighted DLT / UWT step, which is geometrically grounded and has fewer degrees of freedom.

Consequently, v25 can distort 3-D geometry in arbitrary ways to fit the training data, while v80 can mostly only reweight views.

### 3.4 View dropout and variable-view training regularise v80 more effectively

`v46` view dropout forces `v80` to produce reasonable poses even when only 2–4 random views are present.  This acts like a strong structural prior that discourages reliance on any single camera configuration.  `v25` also uses `--use_variable_view_training`, but without the other regularising heads the same augmentation mostly adds noise to the already under-constrained v25 geometry head.

### 3.5 The v25 geometry head has unbounded drift

In `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`:

```python
# Scalar gate initialised to 0.0 gives identity at init; training opens it.
self.residual_scale = nn.Parameter(torch.tensor(0.0))
...
residual = self.fusion_mlp(fused - pred_3d)
return pred_3d + self.residual_scale * residual
```

`residual_scale` is unbounded and has no weight-decay penalty.  Once training proceeds, it can grow large and push the refined 3-D estimate far from the DLT seed, amplifying any overfitting.  The v80 heads, by contrast, predict weights in `(0, 1)` and combine them with a weighted triangulation, so the output stays closer to a geometrically plausible estimate.

### 3.6 Slightly different DLT front end

The v25 script explicitly enables:

```bash
--use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight
```

whereas the v80 script leaves them off.  This is not the main cause of the 33 mm gap, but it matters: v25 relies on a single robust- DLT seed, while v80 lets v45/v52 reweight and re-triangulate repeatedly.  The v80 design is more robust to a poor initial seed because the downstream heads can correct it.

---

## 4. Why neither model beats DLT / Iskakov

Even the better `v80` result (`39.98 mm`) is still behind:

- confidence-weighted DLT: `25.87 mm`
- Iskakov ICCV 2019: `23.35 mm`

The reasons are the same as the overfitting diagnosis:

1. **Training budget mismatch:** The medium run is far too small for the model capacity and task complexity.  The A800 v80 sweep shows that more regularisation (weight decay `1e-4`, lower `lr`) can reach `39.70 mm`, but still not DLT/Iskakov levels.
2. **The task is now genuine pose estimation, not DLT reproduction.**  On circular labels the network only had to approximate DLT; on true GT it must learn a real prior over human pose, camera geometry, and view reliability.
3. **Architecture is not yet tuned for true GT.**  The v80 stack is designed for sparse-view *robustness*, not for maximising full-view accuracy on a single studio dataset.  Its modules add inductive biases that help generalisation, but at the cost of constraining the model away from the true-GT optimum when training data is scarce.

---

## 5. Recommendations to close the gap

The actionable fixes already identified in `docs/v25_divergence_diagnosis.md` apply to both runs:

1. **Increase `train_samples`** to at least 4096, preferably 8192–16384, so the model sees a larger fraction of the ~390 k training frames each epoch.
2. **Add weight decay** (`1e-4` to `2e-4`).  The A800 v80 sweep confirms this is the single most important regulariser.
3. **Enable early stopping** (`--early_stopping_patience 3 --early_stopping_min_delta 0.001`).  Both models reach their best validation within 2–4 epochs; continuing past that only hurts generalisation.
4. **Lower the learning rate** or lengthen the warmup (`lr=5e-4`, `lr_warmup_epochs=2`).
5. **Reduce augmentation intensity** on the small true-GT split (`--outlier_view_prob 0.15`).
6. **Bound the v25 residual gate** (soft clamp or L2 penalty on `residual_scale`) to prevent unbounded drift of the geometry head.
7. **For v80, continue the architecture direction:** the v80/v52/v57 stack is the right robustness story, but it needs more data and stronger regularisation to beat the geometric baselines on a full-view benchmark.

---

## 6. Summary

- **Both v25 and v80 overfit** because the medium recipe (`train_samples=1024`, no weight decay, no early stopping, `lr=1e-3`, strong augmentation) is too small and too lightly regularised for a genuine true-GT pose-estimation task.
- **v80 outperforms v25** because it is a smaller, more constrained model (`817 k` vs `2.73 M` params) surrounded by identity-initialised regularising heads (v45, v46, v50/v51, v52, v80) that reweight views and perform weighted triangulation, rather than freely refining 3-D geometry.
- **Neither beats DLT/Iskakov** under the current medium recipe; doing so requires larger per-epoch training samples, weight decay, early stopping, and possibly a bound on the v25 geometry-head residual gate.
