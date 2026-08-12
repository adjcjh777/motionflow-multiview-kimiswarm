# v53: Reliability-Guided Pose Mixup

**Author:** design-swarm agent  
**Module name:** `reliability_guided_pose_mixup_v53`  
**Status:** Proposal (design-only)  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v52 Uncertainty-Weighted Triangulation, v45 Adaptive Geometry Fusion, v46 Sparse-View Generalization, v51 Cross-Domain Sparse-View Reliability

## 1. Motivation

The v52 Uncertainty-Weighted Triangulation (UWT) module learns per-view/per-joint precision weights and re-triangulates, but it still collapses multi-view evidence into a single 3-D estimate. v53 builds on v52 by explicitly generating a small ensemble of **mixed triangulation hypotheses** from the v52 reliability weights, scoring each hypothesis by geometric consistency, and fusing the best ones. Because the mixup coefficients are derived from v52's learned weights, the module is *reliability-guided*: high-reliability views dominate the candidates, while low-reliability or dropped views are naturally down-weighted. The block is **warm-startable / identity at init**: all candidate multipliers and the residual gate are zero-initialized, and the scoring/fusion MLPs are zero-initialized at their final layers, so a trained v52 checkpoint loaded with v53 enabled produces the same pose.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

The block is inserted **immediately after** the v52 UWT refinement block and **before** the final residual MLP, temporal (v47/v49-Lite), and physical-space alignment (v28/v40) heads:

```
points_2d, K, R, t, features
        |
        v
[earlier fusion: v25/v45 geometry, v46/v51 sparse-view reliability]
        |
        v
UncertaintyWeightedTriangulationV52  ->  pred_3d_gn_uwt, w_uwt
        |
        v
ReliabilityGuidedPoseMixupV53  ->  pred_3d_gn_mix  (B*T, J, 3)
        |
        v
[residual MLP / temporal v47/v49 / physical v28/v40 / losses]
```

### 2.2 Inputs and outputs

```
Inputs                              Outputs
pred_3d        : (B, T, J, 3)  ->  pred_3d_refined : (B, T, J, 3)
points_2d      : (B, T, V, J, 2)   mixup_loss      : scalar
features       : (B, T, V, J, d)   mixup_scores    : (B, T, M, J)  [auxiliary]
K, R, t        : camera params      mixed_weights   : (B, T, M, J)  [auxiliary]
w_uwt          : (B, T, V, J)
view_mask      : (B, T, V)
domain_id      : (B,)
```

### 2.3 Reliability-guided candidate generation

Let `M = v53_rpm_num_candidates`. For each candidate `m`, a learnable scalar mixup multiplier `β_m` and a per-view perturbation `δ_{m,v}` (init `0`) modulate the v52 weights:

```
w_m = sigmoid(β_m + δ_{m,v}) * w_uwt          # (B, T, V, J)
p_m = weighted_dlt_triangulate(points_2d, K, R, t, weights=w_m, view_mask=view_mask)
    # (B, T, J, 3)
```

At initialization, `β_m = 0` and `δ_{m,v} = 0`, so `sigmoid(·) = 0.5` is a constant scale across views and joints. Scaling all v52 weights by the same positive constant leaves the DLT solution unchanged, therefore `p_m = pred_3d` for every `m` — the module is identity at init.

### 2.4 Geometric-consistency scoring

For each candidate, compute the per-view reprojection residual:

```
r_m = || π(p_m) - points_2d ||_2               # (B, T, V, J)
```

The score features are:

```
g_m = concat( p_m - pred_3d,                 # (B, T, J, 3)
              mean_V(r_m), std_V(r_m),       # (B, T, J, 2)
              mean_V(w_m), std_V(w_m) )     # (B, T, J, 2)
```

The per-candidate score is:

```
s_m = MLP_score(g_m)                          # (B, T, J)
```

With `v53_rpm_identity_init=True`, the final layer of `MLP_score` is zero-initialized, so all `s_m = 0` at the start of training.

### 2.5 Reliability-weighted fusion and gated residual

Normalize scores across candidates per joint:

```
α_m = softmax(s_m / v53_rpm_temperature)      # (B, T, M, J)
p_mix = Σ_m α_m * p_m                         # (B, T, J, 3)
```

The fused pose is added as a gated residual to the v52 estimate:

```
Δp = MLP_res(p_mix - pred_3d)                 # final layer zero-init
pred_3d' = pred_3d + v53_rpm_residual_gate * Δp
```

When `v53_rpm_residual_gate = 0.0` and identity initialization is used, `pred_3d' = pred_3d` exactly.

### 2.6 Auxiliary mixup loss

The score network is trained with two signals:

```
target_m = exp( - mean_V(r_m) / 5.0 )         # (B, T, M, J)
L_cons = MSE(softmax(s_m), target_m)
L_ent  = - entropy( α_m )
L_mix  = L_cons + v53_rpm_entropy_weight * L_ent
```

The loss is added to the geometry loss with weight `v53_rpm_loss_weight` only after `v53_rpm_warmup_epochs`.

## 3. Configuration flags

```python
use_v53_reliability_guided_pose_mixup: bool = False
v53_rpm_num_candidates: int = 4
v53_rpm_hidden: int = 64
v53_rpm_n_layers: int = 2
v53_rpm_temperature: float = 0.5
v53_rpm_loss_weight: float = 0.01
v53_rpm_entropy_weight: float = 0.01
v53_rpm_warmup_epochs: int = 0
v53_rpm_identity_init: bool = True
v53_rpm_residual_gate: float = 0.0
v53_rpm_min_weight: float = 0.05
```

## 4. Expected MPJPE impact

| Scenario | Expected delta |
|---|---|
| Sparse 2–3 view evaluation (v46, v51) | −1.0 to −3.0 mm on `MPJPE@2/3` |
| Full-view H36M / MPI-INF-3DHP | −0.2 to −0.8 mm |
| WebBridge / 3DPW actual mode | −0.5 to −1.5 mm |
| Combined with v50 Self-Evolution Feedback Head | up to −2.0 to −3.5 mm on `MPJPE@full` |

Because the block is identity at init, no regression is expected when enabling it on a trained v52 checkpoint before training starts.

## 5. Risks and mitigations

See `docs/swarm_iter27/reports/agent_reliability_guided_pose_mixup_risks.md` for the full register. Top risks include identity-at-init leakage, candidate collapse, extra compute from the candidate ensemble, conflict with the v52 consistency loss, and sparse-view instability.

## 6. 5-step implementation plan

1. **Prototype the standalone module.** Create `motionflow_mv/fusion/reliability_guided_pose_mixup_v53.py` implementing candidate generation, geometric-consistency scoring, reliability-weighted fusion, and the gated residual. Add unit tests for shape correctness, identity-at-init, and gradient flow.

2. **Wire into `OmniMultiViewFusionV5`.** Add the v53 flags to the model constructor. Call the module immediately after `UncertaintyWeightedTriangulationV52.forward`, passing `pred_3d_gn_uwt`, `points_2d`, `features`, the v52 `weights`, `view_mask`, and `domain_id`. Accumulate `mixup_loss` into the geometry loss with weight `v53_rpm_loss_weight` after the warmup period.

3. **Add warm-start smoke tests.** Load a trained v52 checkpoint with `use_v53_reliability_guided_pose_mixup=True` and `v53_rpm_residual_gate=0.0`; assert `val_MPJPE` changes by less than `0.1 mm`. Verify that candidate poses and fused output are identical to the input at initialization.

4. **Run smoke training.** Use a new `configs/benchmark_v53_reliability_guided_pose_mixup_smoke.yaml` with `v53_rpm_loss_weight=0.01` on a small mixed manifest. Compare against the v52 baseline. Target: finite loss, no NaNs, `val_MPJPE` within 3 mm of the v52 baseline after 1 epoch.

5. **Scale to full A800 run.** If smoke passes, add an A800 queue entry in `scripts/launch_v33_a800_queue.py` (e.g. `v53_reliability_guided_pose_mixup_on_v52`) on top of the strongest v52 checkpoint. Report epoch-1 `MPJPE@k`, per-domain metrics, and learned candidate score statistics in the status table.

## 7. Paper story fit

v53 closes the gap between the v52 per-view uncertainty estimation and the downstream physical-space alignment stage of the paper pipeline. By generating and fusing reliability-guided pose hypotheses, it replaces the single triangulation bottleneck with an explicit multi-hypothesis fusion step, reinforcing the narrative: *multi-view video -> human pose extraction -> multi-view fusion and calibration -> physical-space alignment -> optimized motionflow pipeline*.
