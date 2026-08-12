# v54 Outlier-Robust Reliability Refinement (OR2)

**Module name:** `outlier_robust_reliability_v54`  
**Tracking issue:** #184  
**Depends on:** v45 Adaptive Geometry Fusion, v46 Sparse-View Generalization, v51 Cross-Domain Sparse-View Reliability, v52 Uncertainty-Weighted Triangulation, v53 Physical-Space Calibration.

---

## 1. Motivation

v52 Uncertainty-Weighted Triangulation already learns per-view/per-joint precision weights, but its precision MLP is trained end-to-end with MSE and can under-weight clean views or over-weight views that are merely consistent by accident. v53 Physical-Space Calibration corrects the *output* pose against floor/bone constraints, yet it does not explicitly remove grossly inconsistent views before triangulation. The remaining failure mode is **outlier views**: occlusions, motion blur, wrong person association, and cross-domain calibration drift all produce large reprojection residuals that propagate through the fusion pipeline.

v54 OR2 introduces a **learned robust M-estimator** that operates on the residual distribution between the current 3D estimate and each 2D observation. It produces a soft outlier-downweighting factor that refines the v52 UWT weights, then re-triangulates. Because the module is placed *after* v45/v46 and *before or alongside* v52/v53, the entire multi-view fusion block becomes more resilient without changing upstream data loaders.

---

## 2. Architecture

The module is a small residual block with two stages:

### 2.1 Robust residual feature extraction

Given the current triangulated pose `X ∈ R^(B×T×J×3)`, cameras, and 2D keypoints `p_2d ∈ R^(B×T×V×J×2)`, compute the per-view reprojection residual:

```
r_{b,t,v,j} = || Π_v(X_{b,t,j}) - p_{b,t,v,j} ||_2      ∈ R^(B×T×V×J)
```

For each `(v,j)` token we build a feature vector `φ_{v,j}` by concatenating:

* raw residual `r_{v,j}`
* residual rank / MAD statistics across views (median, p25, p75)
* the v52 UWT weight `w_{v,j}^{UWT}`
* optional feature-bias statistics: mean/std of the d-dimensional view token

`φ_{v,j} ∈ R^F`, where `F = 4 + 2 + 1 + 2d` when all branches are enabled.

### 2.2 Learned M-estimator

A compact MLP predicts two quantities:

* `log_scale_{v,j}` — a learnable robust scale in pixel space.
* `logit_γ_{v,j}` — a soft downweighting factor.

The downweight is computed with a Cauchy-like M-estimator:

```
ρ(r; scale) = (scale^2 / 2) · log(1 + (r / scale)^2)
ψ(r; scale)  = r · scale^2 / (r^2 + scale^2)
γ_{v,j}      = sigmoid( -α · (r_{v,j} - β · scale_{v,j}) )
```

`α` is a learned or fixed steepness (default 1.0), and `β` is a learned inlier threshold multiplier (default 2.0).  `γ ∈ [0,1]` is the *outlier-downweight factor*: `γ ≈ 1` for inliers, `γ ≈ 0` for outliers.

The refined triangulation weight is:

```
w_{v,j}^{OR2} = w_{v,j}^{UWT} · γ_{v,j}           if v54_use_uwt_weights
w_{v,j}^{OR2} = γ_{v,j}                           otherwise
```

### 2.3 Re-triangulation and residual correction

`w^{OR2}` is fed into the existing `weighted_dlt_triangulate` utility to obtain `X̃_OR2`.  A final gated residual MLP corrects `X̃_OR2`:

```
ΔX = MLP_res( X̃_OR2 )
g  = sigmoid(gate_init)
X_out = X̃_OR2 + g · ΔX
```

The gate is initialized so `g ≈ 0` at start (e.g. `gate_init = -6.0`), preserving identity at init.  The residual MLP final layer is zero-initialized when `v54_or2_identity_init=True`.

---

## 3. Equations

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `X` | `(B,T,J,3)` | Current 3D pose estimate (from v25/v45/v52/v53) |
| `p_2d` | `(B,T,V,J,2)` | 2D keypoints |
| `K,R,t` | `(B,T,V,3,3)` / `(B,T,V,3)` | Camera intrinsics & extrinsics |
| `r` | `(B,T,V,J)` | Per-view reprojection residual |
| `w^{UWT}` | `(B,T,V,J)` | v52 UWT weights |
| `φ` | `(B,T,V,J,F)` | Residual + feature vector |
| `scale` | `(B,T,V,J)` | Learned robust scale |
| `γ` | `(B,T,V,J)` | Outlier-downweight factor |
| `w^{OR2}` | `(B,T,V,J)` | Refined triangulation weights |
| `X_out` | `(B,T,J,3)` | Output pose |

Core update:

```
r = ||Π(X) - p_2d||
scale, logit_γ = MLP_robust(φ(r, w^{UWT}, F))
γ  = sigmoid(logit_γ)
w^{OR2} = γ ⊙ w^{UWT}
X̃      = weighted_dlt_triangulate(p_2d, K, R, t, w^{OR2})
X_out  = X̃ + sigmoid(gate_init) · MLP_res(X̃)
```

Auxiliary loss (encourages sharp separation of inliers/outliers):

```
L_or2 = - mean( γ_inlier · log γ_inlier + (1-γ_inlier) · log(1-γ_inlier) )
```

where `γ_inlier` is defined from a hard threshold on `r` (e.g. `r < 3 px` from synthetic reprojection noise).

---

## 4. Inputs / Outputs

### Inputs

```python
pred_3d:          torch.Tensor  # (B, T, J, 3)   current 3D estimate
points_2d:        torch.Tensor  # (B, T, V, J, 2)
K:                torch.Tensor  # (B, T, V, 3, 3) or (B, V, 3, 3)
R:                torch.Tensor  # (B, T, V, 3, 3) or (B, V, 3, 3)
t:                torch.Tensor  # (B, T, V, 3)    or (B, V, 3)
feature_tokens:   torch.Tensor  # (B, T, V, J, d)
uwt_weights:      torch.Tensor  # (B, T, V, J)
view_mask:        torch.Tensor  # (B, T, V)  bool/float
```

### Outputs

```python
pred_3d_or2:      torch.Tensor  # (B, T, J, 3)
or2_weights:      torch.Tensor  # (B, T, V, J)
or2_gamma:        torch.Tensor  # (B, T, V, J)
or2_loss:         torch.Tensor  # scalar, optional auxiliary loss
```

---

## 5. Config flags

```python
use_v54_outlier_robust_reliability: bool = False
v54_or2_hidden: int = 64
v54_or2_n_layers: int = 2
v54_or2_weight_type: str = "per_view_joint"   # "per_view_joint" | "per_view" | "per_joint"
v54_or2_estimator: str = "cauchy"            # "cauchy" | "huber" | "geman_mcclure"
v54_or2_scale_init: float = 5.0              # pixels
v54_or2_steepness: float = 1.0               # α in sigmoid
v54_or2_threshold_mult: float = 2.0          # β in sigmoid
v54_or2_use_uwt_weights: bool = True
v54_or2_use_feature_bias: bool = True
v54_or2_recompute_triangulation: bool = True
v54_or2_identity_init: bool = True
v54_or2_min_weight: float = 0.05
v54_or2_loss_weight: float = 0.01
v54_or2_warmup_epochs: int = 0
```

---

## 6. Integration into `OmniMultiViewFusionV5`

Insert v54 **after** v45/v46 view-reliability stages and **around** the v52 UWT block, i.e.:

```
v25/v45 triangulated pose
    -> v46 sparse-view reliability (optional)
    -> v51 CDSVR (optional)
    -> v52 UWT produces w^{UWT}
    -> v54 OR2 refines w^{UWT} -> w^{OR2} and X_out
    -> v53 PSC consumes X_out and w^{OR2} for physical calibration
```

In `omniview_fusion_v5.py` this requires:

1. Add the `use_v54_*` constructor flags.
2. Instantiate `OutlierRobustReliabilityV54(...)` when enabled.
3. In the forward pass, after the v52 block (or replacing its re-triangulation step) call:
   ```python
   pred_3d_or2, or2_weights, or2_gamma, or2_loss = self.outlier_robust_reliability_v54(
       pred_3d=pred_3d, points_2d=points_2d, K=K, R=R, t=t,
       feature_tokens=fused_features, uwt_weights=uwt_weights, view_mask=view_mask
   )
   ```
4. Feed `pred_3d_or2` and `or2_weights` into v53 PSC if PSC is enabled.
5. Add `or2_loss` to the total loss with `v54_or2_loss_weight` after `v54_or2_warmup_epochs`.

---

## 7. Expected MPJPE impact

* **Sparse-view scenarios (2-3 views):** 2-4 mm improvement. Outlier views dominate error when only two or three observations are available; downweighting them sharply improves robustness.
* **Cross-domain 3DPW actual mode:** 1-3 mm improvement. Calibration drift and occlusion create high-residual views; the learned scale adapts to the domain.
* **Full-view WebBridge/H36M:** 0.5-1 mm improvement. Marginal because outliers are already partially averaged out, but sharper inlier selection helps extremities (wrists/ankles).

Conservative combined estimate on A800 full runs: **-0.8 to -2.0 mm** on `val_MPJPE` relative to the v53 baseline.

---

## 8. Risks and mitigations

See `docs/swarm_iter28/reports/agent_outlier_robust_reliability_v54_risks.md` for a detailed risk register.

Top-level concerns: over-aggressive downweighting, scale collapse, and interaction with v52/v53 identity-at-init assumptions.

---

## 9. 5-step implementation plan

1. **Create the module skeleton.** Add `motionflow_mv/fusion/outlier_robust_reliability_v54.py` with `OutlierRobustReliabilityV54`, residual MLP, Cauchy M-estimator, and zero-init / identity-at-init logic. No forward integration yet.

2. **Unit-test the residual math.** Add `tests/test_outlier_robust_reliability_v54.py` that: (a) checks `γ == 1` when `v54_or2_identity_init=True` and no residual MLP is applied, (b) verifies that a synthetic outlier view (`r > 100 px`) gets `γ < 0.1`, and (c) checks re-triangulation output shape `(B,T,J,3)`.

3. **Wire into `OmniMultiViewFusionV5`.** Add flags in `__init__`, instantiate the module, and call it immediately after the v52 UWT block. Route `or2_loss` into the existing auxiliary loss logic alongside `v52_uwt_loss` and `v53_psc_loss`.

4. **Add smoke config and script.** Copy `configs/benchmark_v53_physical_space_calibration_smoke.yaml` to `configs/benchmark_v54_outlier_robust_reliability_smoke.yaml`, enabling `use_v54_outlier_robust_reliability=True` with `v54_or2_loss_weight=0.01`. Add `scripts/run_v54_outlier_robust_reliability_smoke_local_4090.sh`.

5. **Run smoke and verify identity-at-init.** Launch the smoke on RTX 4090. Confirm that loading a v53 checkpoint with v54 enabled changes `val_MPJPE` by ≤ 0.1 mm (identity init preserved). If smoke `val_MPJPE@full` is within 1 mm of the v53 baseline, add the A800 full-run entry to `scripts/launch_v33_a800_queue.py`.
