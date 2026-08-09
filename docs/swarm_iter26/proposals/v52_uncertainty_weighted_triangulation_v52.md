# v52: Uncertainty-Weighted Triangulation (UWT)

**Task identifier:** `design_v52_uncertainty_weighted_triangulation`  
**Status:** Proposal (no code yet)  
**Depends on:** v25 (`MultiViewGeometryFusionV25`), v45 (`adaptive_geometry_fusion`), v46 (`sparse_view_generalization_v46`), v47 (`temporal_aggregation_v47`), v49-Lite, v50 (`SelfEvolutionFeedbackHeadV50`), v51 (`CrossDomainSparseViewReliabilityV51`)

## 1. Motivation

The pipeline triangulates per-view 2D keypoints into a 3D pose, then refines it with v25, v45, v46/v51, v47/v49-Lite, and v50. Each downstream block treats the triangulated pose as a point estimate, so noisy or out-of-domain views already corrupt the 3D point.

**v52** learns a per-view, per-joint precision that drives a differentiable weighted triangulation, down-weighting noisy views *before* the 3D point is formed. The precision predictor and residual MLP are zero-initialized, so without training the module falls back to ordinary triangulation. This matches the paper flow: after per-view pose extraction, fusion must reason about per-view reliability before producing the fused 3D human.

## 2. Module overview

**File:** `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py`

```text
UncertaintyWeightedTriangulationV52(
    d=64, n_views=4, hidden=64, n_layers=2,
    weight_type="per_view_joint",  # per_view_joint | per_view | per_joint
    temperature=1.0, use_geometry_bias=True, use_feature_bias=True,
    identity_init=True, min_weight=0.05, loss_weight=0.01,
)
```

### 2.1 Inputs / outputs

**Forward signature**

```python
pred_3d_ref, triang_dict = uwt(
    feat,         # (B, T, V, J, d)
    points_2d,    # (B, T, V, J, 2)
    conf_2d,      # (B, T, V, J)  [optional]
    K,            # (B, T, V, 3, 3)
    R,            # (B, T, V, 3, 3)
    t,            # (B, T, V, 3)
    pred_3d_init, # (B, T, J, 3)
    view_mask,    # (B, T, V)     [optional]
    domain_id,    # (B, T)        [optional]
)
```

**Outputs**

* `pred_3d_ref`: `(B, T, J, 3)` — refined 3D pose.
* `precision`: `(B, T, V, J)` — per-view per-joint precision before normalization.
* `weights`: `(B, T, V, J)` — normalized triangulation weights, sum over visible views = 1.
* `uwt_loss`: scalar — uncertainty regularization loss.

### 2.2 Architecture and equations

For each view `v` and joint `j` we form a descriptor

```
g_vj = [feat_vj ; ray_dir_vj ; cam_center_v ; conf_2d[v,j] ;
        reproj_residual_vj ; epipolar_score_vj ; feature_norm_vj]
```

A small MLP predicts log-precision:

```
precision_vj = exp(MLP(g_vj) / temperature)
```

The final layer is initialized to zero, so `precision_vj ≈ 1` and the module is a no-op at `t=0`. Masked views are excluded and weights are normalized:

```
w_vj = (precision_vj * view_mask_v) / (sum_v precision_vj * view_mask_v + eps)
```

The weighted 3D point is obtained by weighted DLT:

```
X_j = (A_j^T W_j A_j)^{-1} A_j^T W_j b_j
```

where `A_j` and `b_j` come from the per-view ray formulation. A zero-initialized residual MLP adds a small correction `ΔX`, so `pred_3d_ref = X_j + ΔX`.

### 2.3 Composability with v45/v46/v51

v52 does not replace existing weighting mechanisms; it factors them:

```
effective_weight_vj = w_vj^UWT * w_vj^AGF * r_v^SVG * r_v^CDSVR
```

For stability, v52 is the primary triangulation weight, while v45/v46/v51 weights continue to operate inside downstream refinement blocks.

### 2.4 Uncertainty regularization loss

```
L_uwt = loss_weight * (H_bar(weights) + lambda_cons * mean((pred_3d_ref - pred_3d_init)^2 / precision))
```

`H_bar(weights)` is the mean per-joint negative entropy, discouraging degenerate weight distributions. The second term allows larger deviations where uncertainty is high. `lambda_cons` defaults to `0.1`.

## 3. Integration into `OmniMultiViewFusionV5`

### 3.1 New toggles

```python
use_v52_uncertainty_weighted_triangulation: bool = False,
v52_uwt_hidden: int = 64,
v52_uwt_n_layers: int = 2,
v52_uwt_weight_type: str = "per_view_joint",
v52_uwt_temperature: float = 1.0,
v52_uwt_use_geometry_bias: bool = True,
v52_uwt_use_feature_bias: bool = True,
v52_uwt_identity_init: bool = True,
v52_uwt_min_weight: float = 0.05,
v52_uwt_loss_weight: float = 0.01,
```

### 3.2 Wiring

In `OmniMultiViewFusionV5.__init__`, instantiate the module when the flag is true. In `forward`, call it after the v25/v45 triangulation block and before v46/v47/v48/v49/v50/v51:

```python
if self.uncertainty_weighted_triangulation_v52 is not None:
    pred_3d, uwt_dict = self.uncertainty_weighted_triangulation_v52(...)
    aux_losses["v52_uwt"] = uwt_dict["uwt_loss"]
```

## 4. Expected MPJPE impact

* **Smoke (RTX 4090, 50–100 samples):** ≤ 2 mm change; identity-at-init preserves the baseline.
* **Medium (500–2k samples):** 1–3 mm improvement over v45/v46 by down-weighting noisy views.
* **Full (mixed, 10k+ samples):** 2–5 mm improvement over the strongest v46/v47/v48/v49/v50/v51 stack, especially on `MPJPE@2` and cross-domain evaluation.

## 5. Risks

See `docs/swarm_iter26/reports/agent_uncertainty_weighted_triangulation_v52_risks.md` for detailed risks and mitigations. The main concerns are weight collapse, ill-conditioned weighted DLT gradients, double-counting with v45/v46/v51, memory overhead, and strict identity-at-init.

## 6. Implementation plan

1. **Geometry utilities:** Add batched `weighted_dlt_triangulate` to `motionflow_mv/utils/geometry.py` with `view_mask` support.
2. **Module file:** Implement `UncertaintyWeightedTriangulationV52` with zero-initialized last layers and the loss in §2.4.
3. **Model wiring:** Add the v52 toggle block to `OmniMultiViewFusionV5.__init__` and `forward`, placing it after v45 and before v46/v47/v48/v49/v50/v51.
4. **Smoke test:** Create `configs/benchmark_v52_uwt_smoke.yaml` and `scripts/run_v52_uwt_smoke_local_4090.sh`; verify identity-at-init and uniform weights to within `1e-3`.
5. **Unit tests + ablation:** Add `tests/test_uncertainty_weighted_triangulation_v52.py` covering masked variable views, identity initialization, gradient flow, and ablation against v45 alone.
