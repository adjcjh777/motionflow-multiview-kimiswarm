# v53: Bone-Length-Aware Fusion on Uncertainty-Weighted Triangulation

**Author:** design-swarm agent  
**Module name:** `bone_length_aware_fusion_v53`  
**Status:** Proposal (design-only)  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v52 Uncertainty-Weighted Triangulation, v45 Adaptive Geometry Fusion, v46 Sparse-View Generalization, v51 Cross-Dataset Sparse-View Reliability

## 1. Motivation

The v52 Uncertainty-Weighted Triangulation (UWT) module improves per-joint triangulation by learning per-view/per-joint precision weights, but it still treats joints independently during the re-triangulation and residual refinement steps. Bone lengths are a strong, domain-invariant anthropometric prior: for a given subject in a short clip, most bone lengths are nearly constant. v53 builds directly on the v52 UWT foundation by adding a **bone-length-aware fusion** stage that takes the UWT-refined pose, computes per-bone length corrections, and applies a kinematic-chain offset before the temporal (v47/v49) and physical (v28/v40) heads. The module is **warm-startable / identity at init**: the final projection layers are zero-initialized and the residual gate defaults to `0.0`, so a trained v52 checkpoint loaded with v53 enabled produces the same pose.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

The block is inserted **immediately after** the v52 UWT refinement block and **before** the temporal aggregation (v47/v49-Lite) and physical-space alignment (v28/v40) heads:

```
points_2d, K, R, t, features
        |
        v
[earlier fusion: v25/v45 geometry, v46/v51 sparse-view reliability]
        |
        v
pred_3d_gn  (B*T, J, 3)
        |
        v
UncertaintyWeightedTriangulationV52  ->  pred_3d_gn_uwt  (B*T, J, 3)
        |
        v
BoneLengthAwareFusionV53  ->  pred_3d_gn_bl  (B*T, J, 3)
        |
        v
[temporal v47/v49, physical v28/v40, losses]
```

### 2.2 Inputs and outputs

```
Inputs                              Outputs
pred_3d        : (B, T, J, 3)  ->  pred_3d_refined : (B, T, J, 3)
points_2d      : (B, T, V, J, 2)   bone_loss       : scalar
weights        : (B, T, V, J)       bone_lengths    : (B, T, num_bones)  [auxiliary]
view_mask      : (B, T, V)
parents        : (J,)
```

### 2.3 Bone extraction and uncertainty encoding

For each frame and each bone `j` with parent `parent(j)`:

```
b_j(t)  = pred_3d[:, t, j] - pred_3d[:, t, parent(j)]   # (B, 3)
l_j(t)  = || b_j(t) ||_2                                  # (B,)
```

A bone is valid only if both endpoint joints are visible in at least `v53_bone_min_visible_views` views. Visibility is derived from the existing `view_mask` and the per-joint visibility tensor already in scope.

The v52 per-joint log-precision `log_precision` is pooled over views to obtain a per-joint uncertainty:

```
σ_j^{-1} = mean_v log_precision[:, :, v, j]   # (B, T, J)
```

Each bone is encoded as:

```
e_j = MLP_bone( concat(
    log(l_j + ε),
    log σ_j^{-1},
    log σ_{parent(j)}^{-1},
    bone_type_embed[j]
) )   # (B, T, d_bone)
```

### 2.4 Bone-length consistency network

A lightweight transformer encoder processes the encoded bone tokens:

```
h_bone = TransformerBoneEncoder(e)   # (B, T, num_bones, d_bone)
```

Two heads predict an additive length correction and a per-bone gate:

```
Δl_j     = MLP_length(h_bone_j)          # zero-initialized final layer
γ_j      = sigmoid( MLP_gate(h_bone_j) ) # near-zero init (gate ~ 0)
l'_j     = l_j + v53_bone_residual_gate * γ_j * Δl_j
```

The corrected bone lengths are converted back to per-joint 3-D offsets through a single-layer skeleton Graph Network that propagates the length residuals along the kinematic tree:

```
Δp_j    = GNN_skeleton( {l'_k - l_k}, parents )   # zero-initialized output
pred_3d' = pred_3d + v53_bone_residual_gate * Δp_j
```

When `v53_bone_residual_gate = 0.0`, the block is exactly identity. When `v53_bone_identity_init = True`, the final `MLP_length` and `GNN_skeleton` output layers are zero-initialized.

### 2.5 Auxiliary bone-length prior loss

A learnable mean `μ_j` and variance `σ_j^2` per bone define a soft Gaussian prior:

```
L_bone = Σ_j w_j * [ 0.5 * (l'_j - μ_j)^2 / σ_j^2 + 0.5 * log(2π σ_j^2) ]
```

`w_j` down-weights low-visibility or uncertain bones using the v52 per-joint uncertainty. The loss is added to the total training objective with weight `v53_bone_loss_weight` only after `v53_bone_loss_warmup_epochs`.

## 3. Configuration flags

```python
use_bone_length_aware_fusion_v53: bool = False
v53_bone_hidden: int = 64
v53_bone_n_layers: int = 2
v53_bone_n_heads: int = 4
v53_bone_dropout: float = 0.1
v53_bone_loss_weight: float = 0.01
v53_bone_loss_warmup_epochs: int = 0
v53_bone_residual_gate: float = 0.0
v53_bone_identity_init: bool = True
v53_bone_min_visible_views: int = 2
v53_bone_use_uncertainty: bool = True
```

## 4. Expected MPJPE impact

| Scenario | Expected delta |
|---|---|
| Sparse 2–3 view evaluation (v46, v51) | −1.5 to −3.5 mm on `MPJPE@2/3` |
| Full-view H36M / MPI-INF-3DHP | −0.3 to −1.0 mm |
| WebBridge / 3DPW actual mode | −0.8 to −2.0 mm |
| Combined with v50 Self-Evolution Feedback Head | up to −2.5 to −4.0 mm on `MPJPE@full` |

Because the block is identity at init, no regression is expected when enabling it on a trained v52 checkpoint before training starts.

## 5. Risks and mitigations

See `docs/swarm_iter27/reports/agent_bone_length_aware_fusion_v53_risks.md` for the full register. Top risks include warm-start leakage, canonical-skeleton collapse, conflict with the physical-space alignment losses, cross-dataset skeleton mismatch, and sparse-view over-constraint.

## 6. 5-step implementation plan

1. **Prototype the standalone module.** Create `motionflow_mv/fusion/bone_length_aware_fusion_v53.py` implementing bone extraction, uncertainty pooling from v52 log-precision, transformer-based length/gate heads, skeleton GNN back-projection, and the Gaussian bone-length prior loss. Add unit tests for shape correctness, identity-at-init, and gradient flow.

2. **Wire into `OmniMultiViewFusionV5`.** Add the v53 flags to the model constructor. Call the module immediately after `UncertaintyWeightedTriangulationV52.forward`, passing `pred_3d_gn_uwt`, `points_2d`, the v52 `weights`, `view_mask`, and the runtime parent list. Accumulate `bone_loss` into the geometry loss with weight `v53_bone_loss_weight` after the warmup period.

3. **Add warm-start smoke tests.** Load a trained v52 checkpoint with `use_bone_length_aware_fusion_v53=True` and `v53_bone_residual_gate=0.0`; assert `val_MPJPE` changes by less than `0.1 mm`. Gradually increase the gate and verify the bone-length loss decreases and the pose remains plausible.

4. **Run smoke training.** Use `configs/benchmark_v53_bone_length_aware_fusion_smoke.yaml` with `v53_bone_loss_weight=0.01` on a small mixed manifest. Compare against the v52 baseline. Target: finite loss, no NaNs, val_MPJPE within 3 mm of the v52 baseline after 1 epoch.

5. **Scale to full A800 run.** If smoke passes, add an A800 queue entry in `scripts/launch_v33_a800_queue.py` (e.g. `v53_bone_length_aware_fusion_on_v52`) on top of the strongest v52 checkpoint. Report epoch-1 `MPJPE@k`, per-domain metrics, and learned bone-length prior statistics in the status table.

## 7. Paper story fit

v53 closes the loop between the v52 uncertainty-weighted triangulation and the physical-space alignment stage of the paper pipeline. By injecting a lightweight, learnable skeleton prior directly into the multi-view fusion output, it turns raw per-joint triangulations into kinematically consistent 3-D poses before temporal and physical refinement, reinforcing the narrative: *multi-view video -> human pose extraction -> multi-view fusion and calibration -> physical-space alignment -> optimized motionflow pipeline*.
