# v54: Reliability-Guided Pose Mixup

**Author:** design-swarm agent  
**Module name:** `reliability_guided_pose_mixup_v54`  
**Status:** Proposal (design-only)  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v52 Uncertainty-Weighted Triangulation, v53 Physical-Space Calibration, v51 Cross-Domain Sparse-View Reliability

## 1. Motivation

The v53 Physical-Space Calibration (PSC) block already refines the v52 triangulated pose with floor height, bone-length, and reprojection constraints. v54 goes one step further: it treats the calibrated pose as one hypothesis and mixes it, on a **per-joint** basis, with a learned canonical-pose anchor. The mixing weights are conditioned on the v52 per-joint reliability so that low-confidence joints are pulled toward a stable reference, while high-confidence joints stay close to the calibrated estimate. The block is inserted **after v53** and is **identity-at-init**, so loading a trained v53 checkpoint with v54 enabled does not change the forward output until training starts.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

```
points_2d, K, R, t, features
        |
        v
[ earlier fusion: v25/v45, v46/v51, v47/v49, v52 UWT ]
        |
        v
PhysicalSpaceCalibrationV53  ->  pred_3d_psc, psc_floor, psc_bone_scale
        |
        v
ReliabilityGuidedPoseMixupV54  ->  pred_3d_v54  (B*T, J, 3)
        |
        v
[ final output / residual MLP / losses ]
```

### 2.2 Inputs and outputs

```
Inputs                              Outputs
pred_3d        : (B, T, J, 3)  ->  pred_3d_refined : (B, T, J, 3)
reliability    : (B, T, J)     ->  mixup_loss      : scalar
uncertainty    : (B, T, J)     ->  mixup_alpha     : (B, T, J)
domain_id      : (B,)          ->  residual_norm   : (B, T, J) [aux]
```

`reliability` is the maximum per-joint v52 UWT weight aggregated over visible views; `uncertainty` is the corresponding log-precision. These are already available in the v52 forward pass.

### 2.3 Learned canonical anchor

Maintain a learnable parameter

```
P_anchor ∈ R^(J, 3)
```

initialized from a small held-out mean pose (or zeros if no mean is available). During training it is updated by back-propagation; at inference it is fixed. A per-joint scale vector `s_j` (init `1.0`) is also learned so the anchor can adapt to subject scale without changing the identity property.

### 2.4 Reliability-conditioned mixing

For each joint `j`, build a feature vector

```
g_j = concat(
    pred_j,                # (3)
    reliability_j,         # (1)
    uncertainty_j,         # (1)
    domain_embed(domain_id) # (D)
)  # (5 + D)
```

A ross-joint transformer encoder (2 layers, `v54_rgpm_num_heads` heads, hidden `v54_rgpm_hidden`) processes tokens of shape `(B, T, J, hidden)` and outputs per-joint mixing logits and a residual correction. The mixing weight and gate are

```
α_j = sigmoid( MLP_mix(g_j) )                # (B, T, J)
β   = sigmoid( v54_rgpm_residual_gate_init )  # scalar, init ≈ 0
Δ_j = MLP_res(g_j)                           # (B, T, 3), final layer zero-init
```

The refined pose is

```
mixed_j = (1 - α_j) * pred_j + α_j * (s_j * P_anchor_j)
pred_j' = pred_j + β * Δ_j                  # residual only, no anchor
```

or, in closed form for the final output,

```
pred_j^{v54} = (1 - α_j) * pred_j + α_j * (s_j * P_anchor_j) + β * Δ_j
```

At initialization `α_j ≈ 0`, `β ≈ 0`, and `Δ_j = 0`, so `pred_j^{v54} = pred_j` exactly.

### 2.5 Auxiliary loss

The block is trained with a reliability-aware consistency loss that penalizes large deviations from the calibrated pose only where the calibrated estimate is already confident:

```
L_v54 = (1 / J) Σ_j reliability_j * || pred_j^{v54} - pred_j ||_2^2
        + λ_floor * floor_loss(pred^{v54})
        + λ_bone * bone_length_loss(pred^{v54})
```

`reliability_j` acts as a soft attention mask: the module is not allowed to "explain away" a high-confidence joint. The floor and bone regularizers are optional, controlled by `v54_rgpm_floor_weight` and `v54_rgpm_bone_weight`, and reuse the v28/v40 helpers already present in the model. The loss is added to the geometry loss with weight `v54_rgpm_loss_weight` only after `v54_rgpm_warmup_epochs`.

## 3. Configuration flags

```python
use_v54_reliability_guided_pose_mixup: bool = False
v54_rgpm_hidden: int = 64
v54_rgpm_n_layers: int = 2
v54_rgpm_num_heads: int = 4
v54_rgpm_dropout: float = 0.1
v54_rgpm_loss_weight: float = 0.01
v54_rgpm_floor_weight: float = 0.0
v54_rgpm_bone_weight: float = 0.0
v54_rgpm_warmup_epochs: int = 0
v54_rgpm_identity_init: bool = True
v54_rgpm_residual_gate_init: float = -6.0
v54_rgpm_anchor_init: str = "zero"          # "zero" or "mean_pose"
v54_rgpm_use_domain_conditioning: bool = True
v54_rgpm_min_alpha: float = 0.0
v54_rgpm_max_alpha: float = 0.5
```

## 4. Expected MPJPE impact

| Scenario | Expected delta |
|---|---|
| Full-view H36M / MPI-INF-3DHP | −0.2 to −0.7 mm on `MPJPE@full` |
| Sparse/variable-view (MPJPE@2/MPJPE@3) | −1.0 to −2.5 mm |
| Cross-domain 3DPW / WebBridge actual mode | −0.8 to −1.8 mm |
| Stacked with v50 SEFH + v51 CDSVR | up to −1.5 to −3.0 mm on `MPJPE@full` |

Because the block is identity-at-init, no regression is expected when enabling it on a trained v53 checkpoint before the first training step.

## 5. Risks and mitigations

See `docs/swarm_iter28/reports/agent_reliability_guided_pose_mixup_v54_risks.md` for the full register.

## 6. 5-step implementation plan

1. **Prototype the standalone module.** Create `motionflow_mv/fusion/reliability_guided_pose_mixup_v54.py` implementing the learned anchor, cross-joint transformer, reliability-conditioned mixing, gated residual, and auxiliary loss. Add unit tests for output shape, identity-at-init, and gradient flow.

2. **Wire into `OmniMultiViewFusionV5`.** Add the v54 flags to the model constructor. Call the module immediately after `PhysicalSpaceCalibrationV53.forward`, passing `pred_3d_psc`, the per-joint v52 reliability/uncertainty, and `domain_id`. Accumulate `mixup_loss` into the geometry loss with weight `v54_rgpm_loss_weight` after the warmup period.

3. **Add warm-start smoke tests.** Load a trained v53 checkpoint with `use_v54_reliability_guided_pose_mixup=True` and `v54_rgpm_residual_gate_init=-6.0`; assert `val_MPJPE` changes by less than `0.1 mm`. Verify that `pred_3d_v54 == pred_3d_psc` at initialization.

4. **Run smoke training.** Create `configs/benchmark_v54_reliability_guided_pose_mixup_smoke.yaml` with `v54_rgpm_loss_weight=0.01` on a small mixed manifest. Compare against the v53 baseline. Target: finite loss, no NaN, `val_MPJPE` within 2 mm of the v53 baseline after 1 epoch.

5. **Scale to full A800 run.** If smoke passes, add an A800 queue entry in `scripts/launch_v33_a800_queue.py` (e.g. `v54_reliability_guided_pose_mixup_on_v53`) on top of the strongest v53 checkpoint. Report epoch-1 `MPJPE@k`, per-domain metrics, and the mean/median learned `mixup_alpha` in the status table.

## 7. Paper story fit

v54 closes the loop between v52/v53 fusion/calibration and the final optimized MotionFlow pipeline. By reliability-guiding the blending of the calibrated pose with a learned canonical anchor, the module provides a principled, physically-aware refinement step that is still identity-at-init. This reinforces the paper narrative: *multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline*.
