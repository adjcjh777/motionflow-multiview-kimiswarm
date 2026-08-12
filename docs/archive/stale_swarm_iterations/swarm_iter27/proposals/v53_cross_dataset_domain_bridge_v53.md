# v53 — Cross-Dataset Domain Bridge (CDDB-v53)

**Tracking issue:** `#187` (proposed)  
**Status:** Proposal / design-swarm  
**Depends on:** v25, v45-AGF, v46-SVG, v47/v49-temporal, v48-domain, v49-Lite, v50-SEFH, v51-CDSVR, **v52-UWT**

## 1. Motivation

The v52 Uncertainty-Weighted Triangulation (UWT) module produces per-view, per-joint precision weights `w_uwt ∈ R^{B×T×V×J}` and a refined triangulated pose `pred_3d_uwt ∈ R^{B×T×J×3}`. While v52 improves triangulation robustness, its weights and residual correction are still learned from a mixed-domain feature distribution. When a source-domain model is evaluated on 3DPW or MPI, the learned uncertainty patterns transfer imperfectly because each dataset has a distinct signature in camera rig, body proportions, and background clutter. v53 adds a **pose-level cross-dataset domain bridge** that re-calibrates the v52 triangulated pose into a domain-invariant canonical skeleton space, explicitly closing the gap between *multi-view fusion/calibration* and *physical-space alignment* in the paper pipeline.

```
multi-view video
    ↓
human pose extraction (backbone 2D → 3D lift)
    ↓
multi-view fusion + calibration (v25, v45, v46, v52-UWT)
    ↓
Cross-Dataset Domain Bridge v53  ← new: domain-agnostic pose re-calibration
    ↓
physical-space alignment (v28 / v40)
    ↓
optimized MotionFlow pipeline
```

## 2. Architecture

`CrossDatasetDomainBridgeV53` is inserted **after** v52-UWT and before the physical-space alignment loss. It operates on the triangulated 3D pose plus the UWT weights, is warm-startable/identity-at-init, and returns a refined pose plus an auxiliary bridge loss.

### 2.1 High-level block

```
Input:
  pred_3d_uwt  ∈ R^{B×T×J×3}
  uwt_weights  ∈ R^{B×T×V×J}
  x            ∈ R^{B×T×V×J×d}
  domain_label ∈ {0,…,D-1}^{B}
    │
    ▼
[Uncertainty-Guided Skeleton Tokens]
    │
    ▼
[Domain-Conditional FiLM]  (identity at init)
    │
    ▼
[Cross-Domain Pose Prototype Attention]
    │
    
[Canonical Pose Residual MLP]  (zero-init final layer)
    │
    ▼
[Gated Residual Update]  α ≈ 0.0025 at init
    │
    ▼
Output: pred_3d_bridge ∈ R^{B×T×J×3}, bridge_loss
```

### 2.2 Component details

**Uncertainty-Guided Skeleton Tokens.** For each view `v` and joint `j`, concatenate the per-view feature, the log UWT weight, and the per-joint reprojection residual norm `r`:

```
t_{b,t,v,j} = concat( x_{b,t,v,j},
                      log(uwt_weights_{b,t,v,j} + 1e-6),
                      r_{b,t,v,j} )   ∈ R^{d+2}
```

The residual `r_{b,t,v,j} = ||p_{b,t,j} - π_v^{-1}(·)||_2` is computed from the current triangulated pose; it is treated as a fixed geometry cue, not back-propagated through the pose.

**Domain-Conditional FiLM.** A domain embedding projects to per-channel affine parameters `(γ, β)`:

```
h = LayerNorm(t)
(γ, β) = FiMLP(domain_label)
h̃ = γ ∘ h + β
gamma, beta initialized so that γ=1, β=0 at init
```

**Cross-Domain Pose Prototype Attention.** Maintain a learnable prototype bank `P ∈ R^{D×J×k}` where `D` is the number of domains and `k` a latent dimension. Per-joint query tokens are pooled over views and time windows:

```
q_j = mean_pool_{t,v}(h̃_{*,*,v,j}) W_q      ∈ R^{B×k}
K   = P W_k                                  ∈ R^{D×J×k}
V   = P W_v                                  ∈ R^{D×J×k}
A_j = softmax_j( q_j K_j^T / sqrt(k) )       ∈ R^{B×D}
z_j = A_j V_j                                ∈ R^{B×k}
```

Only joint `j` attends to the `j`-th prototype column; attention is over domains, not joints. At initialization `P` is set to the mean per-domain pose token from a small source batch, so `z_j ≈ q_j` and the module is near-identity.

**Canonical Pose Residual & Gated Update.** A two-layer MLP maps the prototype-blended token to a 3D residual. Its final layer is zero-initialized, and a scalar gate `α = sigmoid(g)` is initialized to `sigmoid(-6.0) ≈ 0.0025`:

```
Δpred = MLP(z)              ∈ R^{B×T×J×3}
pred_3d_bridge = pred_3d_uwt + α · Δpred
```

### 2.3 Auxiliary bridge losses

```
L_bridge = v53_cdb_lambda_pose · L_pose + v53_cdb_lambda_adv · L_adv + v53_cdb_lambda_bone · L_bone
```

- **Pose consistency** `L_pose`: cross-domain L2 between the gated output and the pseudo-ground-truth 3D pose available during training.
- **Domain adversarial loss** `L_adv`: a tiny GRL discriminator (`v48` style) on the pooled canonical token `mean_{T,V,J}(h̃)` is trained to predict `domain_label`; the encoder is trained to fool it, pushing the latent space domain-invariant.
- **Bone-length regularizer** `L_bone`: enforces that the bone lengths of `pred_3d_bridge` are stable across domains by minimizing variance of canonical bone lengths within a mini-batch.

## 3. Inputs and outputs

| Symbol | Tensor shape | Description |
|--------|--------------|-------------|
| `pred_3d_uwt` | `(B, T, J, 3)` | 3D pose after v52-UWT re-triangulation |
| `uwt_weights` | `(B, T, V, J)` | v52 UWT per-view triangulation weights |
| `x` | `(B, T, V, J, d)` | Per-view fused feature tokens from v51-CDSVR |
| `domain_label` | `(B,)` | Integer domain id (WebBridge=0, H36M=1, MPI=2, 3DPW=3, ...) |
| `cam_params` | `dict` | `{K, R, t}` for reprojection residual computation |
| `view_mask` | `(B, T, V)` | Binary mask for missing/dropped views |

**Outputs:**

| Symbol | Tensor shape | Description |
|--------|--------------|-------------|
| `pred_3d_bridge` | `(B, T, J, 3)` | Domain-agnostic refined 3D pose |
| `bridge_loss` | `scalar` | Weighted pose + adversarial + bone-length loss |

## 4. Config flags

```yaml
# v53 cross-dataset domain bridge
use_v53_cross_dataset_domain_bridge: false
v53_cdb_hidden: 64
v53_cdb_n_heads: 4
v53_cdb_n_layers: 1
v53_cdb_dropout: 0.1
v53_cdb_num_prototypes: 32          # k in the cross-domain prototype bank
v53_cdb_pose_loss_weight: 1.0
v53_cdb_adv_loss_weight: 0.05
v53_cdb_bone_loss_weight: 0.001
v53_cdb_temperature: 0.1
v53_cdb_identity_gate_init: -6.0
v53_cdb_use_film: true
v53_cdb_use_grl_discriminator: true
v53_cdb_use_bone_regularizer: true
v53_cdb_warmup_epochs: 0
```

## 5. Expected MPJPE impact

| Scenario | Expected change | Rationale |
|----------|-----------------|-----------|
| Source-only H36M/WebBridge | ±0.5 mm | Identity gate preserves v52 baseline; small regularization may help |
| Cross-dataset H36M→MPI | -2 to -4 mm | Domain-invariant pose space reduces latent distribution shift |
| Cross-dataset H36M→3DPW | -4 to -7 mm | Pose-level bridge explicitly targets the target-domain gap |
| Variable-view (2 views) | -2 to -4 mm @ MPJPE@2 | UWT weights inform stable canonical tokens when views drop |

Acceptance criteria for smoke (RTX 4090, d=64, 1000 samples/2 epochs):

- `val_MPJPE` within 2 mm of the v52-UWT baseline on the same config.
- No NaNs or runaway losses in `bridge_loss`.
- Per-domain validation MPJPE on 3DPW is lower than the v48-domain and v52-UWT baselines.

## 6. Implementation plan

1. **Module scaffold** (`motionflow_mv/fusion/cross_dataset_domain_bridge_v53.py`): implement `CrossDatasetDomainBridgeV53` with uncertainty-guided skeleton tokens, FiLM, cross-domain pose-prototype attention, zero-initialized residual MLP, and gated residual update.
2. **Integration into `OmniMultiViewFusionV5`**: add the constructor block immediately after the v52-UWT call (around line 1748), gated by `use_v53_cross_dataset_domain_bridge`. Pass `pred_3d_uwt`, `uwt_weights`, fused features, and `domain_id` into the module; replace the downstream `pred_3d` with `pred_3d_bridge`.
3. **Trainer loss hook**: in `experiments/train_omniview_fusion_v5_webbridge_multi.py`, add `total_loss = pose_loss + v53_cdb_pose_loss_weight * bridge_loss + …` only when the flag is on and `epoch >= v53_cdb_warmup_epochs`.
4. **Smoke config**: create `configs/benchmark_v53_cross_dataset_domain_bridge_smoke.yaml` mirroring `v52_uwt_smoke` but with `use_v53_cross_dataset_domain_bridge: true` and 3DPW included as a validation domain.
5. **Evaluation**: extend `experiments/eval_variable_views.py` to report `MPJPE@full`, `MPJPE@3`, and `MPJPE@2` broken down by domain (H36M, MPI, 3DPW, WebBridge) so the bridge benefit can be isolated.

## 7. Relation to existing modules

- **v48 domain generalization**: v48 adapts batch statistics at the feature level; v53 performs pose-level domain bridging on top of v52-UWT outputs.
- **v51 CDSVR**: v51 produces per-(view,joint) reliability; v53 consumes these features and the v52 weights to build a domain-agnostic canonical pose.
- **v52 UWT**: v53 directly refines the v52 triangulated pose and is warm-startable from any v52 checkpoint.
