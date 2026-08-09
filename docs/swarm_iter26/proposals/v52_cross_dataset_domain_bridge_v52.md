# v52 — Cross-Dataset Domain Bridge

**Tracking issue:** `#183` (proposed)  
**Status:** Proposal / design-swarm  
**Depends on:** v45-AGF, v46-SVG, v48-domain, v51-CDSVR

## 1. Motivation

The current pipeline (`v25` geometry fusion → `v30` hierarchical multiview → `v45` adaptive geometry fusion → `v46` sparse-view generalization → `v47/v49` temporal → `v48` domain adapter → `v50` SEFH → `v51` CDSVR) stacks domain-adaptation components, but each module still consumes features that carry a strong domain-specific signature. When training on WebBridge + H36M + MPII and testing on 3DPW, the latent distribution of per-view joint tokens shifts because camera rigs, actor body proportions, and background statistics differ. The result is that sparse-view reliability, triangulation weights, and physical-space alignment are computed on a representation that is not domain-neutral, limiting cross-dataset generalization.

**Paper alignment:** The proposal explicitly bridges the gap between *multi-view fusion/calibration* and *physical-space alignment* by injecting a domain-invariant latent space before the final pose head. It turns the pipeline into:

```
multi-view video
    ↓
human pose extraction (backbone 2D -> 3D lift)
    ↓
multi-view fusion + calibration (v25, v30, v45, v46)
    ↓
Cross-Dataset Domain Bridge (v52)  ← new
    ↓
physical-space alignment (v28 / v40)
    ↓
optimized MotionFlow pipeline
```

## 2. Architecture

`CrossDatasetDomainBridgeV52` is a small transformer block that operates on the per-view joint tokens produced by the preceding fusion stages. It learns a **shared, geometry-preserving latent space** while keeping the network warm-startable/identity-at-init.

### 2.1 High-level block

```
Input x  ∈ R^{B×T×V×J×d}
    │
    ▼
[LayerNorm]
    │
    ▼
[Domain-Conditional MLP (FiLM)]  ← uses domain_label, warm-start to identity
    │
    ▼
[Cross-Domain Joint Attention]     ← attention across domain prototypes
    │
    ▼
[Geometry-Preserving Residual Gate]  ← α ≈ 0 at init
    │
    ▼
Output x_bridge  ∈ R^{B×T×V×J×d}
```

### 2.2 Component details

**Domain-Conditional FiLM.** Given input `x` and a domain index `domain_label[b] ∈ {0, …, D-1}`, compute

```
h = LayerNorm(x)
(γ, β) = FiLM(domain_label)
h̃ = γ ∘ h + β
```

`FiLM: Z^D → R^{2d}` is an lookup embedding plus a 1-layer MLP. At initialization the MLP outputs zeros and the embedding is zero-centered, so `γ = 1`, `β = 0`, and `h̃ = h`.

**Cross-Domain Joint Attention (CDJA).** Per-joint tokens are refined by attending to a small set of **domain-prototype tokens** shared across all batches:

```
P ∈ R^{D×J×k}            learned domain-prototype bank (k = 32)
Q  = h̃ W_q              (B,T,V,J,d)
K  = P W_k              (D,J,k → D,J,d)
V  = P W_v              (D,J,d)
A  = softmax(Q K^T / sqrt(d))   over (D,J)
h' = A V
```

Only the **same joint index** attends to its own prototype row, so `A` is over `D` domains, not over joints. The prototype bank is initialized from the mean of an initial forward pass over a small source batch, giving a near-identity start.

**Geometry-Preserving Residual Gate.** The final output is a convex interpolation between the original and transformed features:

```
α = sigmoid(g)          g initialized to -6.0  ⇒ α ≈ 0.0025
x_bridge = (1 - α) · x + α · h'
```

Because `h' = h = x` in expectation at initialization (prototypes initialized to input means and FiLM is identity), the module is warm-startable and does not perturb the already-good v45/v46/v51 features at epoch 0.

### 2.3 Auxiliary bridge loss

A joint-level contrastive loss aligns the same joint across domains and pushes different joints apart:

```
z = LayerNorm(x_bridge)            # (B,T,V,J,d)
z_bj = mean_pool over T,V          # (B,J,d)

For each joint j:
    L_pos   =  -log( exp(sim(z_{b,j}, z_{b',j}) / τ) / Σ_{j'} exp(sim(z_{b,j}, z_{b',j'}) / τ) )

L_bridge = mean_j L_pos
sim(a,b) = a^T b / (||a|| ||b||)
```

where `b, b'` are two different domains present in the same mini-batch. The loss weight is `v52_cdb_bridge_loss_weight` (default `0.005`).

To preserve the geometric/physical structure of the original pipeline, `L_bridge` is optionally augmented with a bone-length consistency term:

```
L_bone = Σ_{(j_parent, j_child)} | ||z_j - z_parent||_2 - μ_bone |^2
```

using the canonical 17-joint skeleton. The total auxiliary loss is `L_aux = v52_cdb_bridge_loss_weight · L_bridge + v52_cdb_bone_loss_weight · L_bone`.

## 3. Inputs and outputs

| Symbol | Tensor shape | Description |
|--------|--------------|-------------|
| `x` | `(B, T, V, J, d)` | Multi-view fused joint features from v45/v46/v51 |
| `domain_label` | `(B,)` | Integer domain id (WebBridge=0, H36M=1, MPI=2, 3DPW=3, ...) |
| `view_mask` | `(B, V)` | Binary mask for missing/dropped views |
| `cam_params` | `dict` | `{K, R, t}` intrinsics/extrinsics for optional geometry regularization |

**Outputs:**

| Symbol | Tensor shape | Description |
|--------|--------------|-------------|
| `x_bridge` | `(B, T, V, J, d)` | Domain-invariant features, plug-compatible with existing head |
| `bridge_loss` | `scalar` | Auxiliary contrastive + bone-length loss (zero at init, weighted) |

## 4. Config flags

```yaml
# v52 cross-dataset domain bridge
use_v52_cross_dataset_domain_bridge: false
v52_cdb_hidden: 64
v52_cdb_n_heads: 4
v52_cdb_n_layers: 1
v52_cdb_dropout: 0.1
v52_cdb_num_prototypes: 32          # k in the prototype bank
v52_cdb_bridge_loss_weight: 0.005
v52_cdb_bone_loss_weight: 0.001
v52_cdb_temperature: 0.1
v52_cdb_identity_gate_init: -6.0
v52_cdb_use_geometry_regularizer: true
v52_cdb_use_film: true
v52_cdb_use_prototype_init: true    # initialize prototypes from source batch mean
```

## 5. Expected MPJPE impact

| Scenario | Expected change | Rationale |
|----------|---------------|-----------|
| Source-only H36M/WebBridge | ±0.5 mm | Identity gate keeps baseline intact; small regularization may help |
| Cross-dataset H36M→MPI | -2 to -4 mm | Domain-invariant features reduce distribution shift in sparse views |
| Cross-dataset H36M→3DPW | -3 to -6 mm | Bridge loss explicitly aligns target-domain latent geometry with source |
| Variable-view (2 views) | -2 to -3 mm @ MPJPE@2 | Prototype attention provides stable features when view count drops |

Acceptance criteria for smoke (RTX 4090, d=64, 1000 samples/2 epochs):

- `val_MPJPE` within 2 mm of the v51-CDSVR baseline on the same config.
- No NaNs or runaway losses in `bridge_loss`.
- Per-domain validation MPJPE on 3DPW is lower than the v48-domain baseline.

## 6. Implementation plan

1. **Module scaffold** (`motionflow_mv/fusion/cross_dataset_domain_bridge_v52.py`): implement `CrossDatasetDomainBridgeV52` with FiLM, prototype bank, cross-domain attention, and residual gate. Initialize the gate to `sigmoid(-6.0)` and the output projection to near-zero.
2. **Integration into `OmniMultiViewFusionV5`**: add the constructor block after v51-CDSVR (around line 988), gated by `use_v52_cross_dataset_domain_bridge`. Wire the returned `bridge_loss` into the trainer loss dictionary.
3. **Trainer loss hook**: in `experiments/train_omniview_fusion_v5_webbridge_multi.py`, add `total_loss = pose_loss + v52_cdb_bridge_loss_weight * bridge_loss` only when the flag is on and `epoch >= v52_cdb_warmup_epochs`.
4. **Smoke config**: create `configs/benchmark_v52_cross_dataset_domain_bridge_smoke.yaml` mirroring `v51_cdsvr_smoke` but with `use_v52_cross_dataset_domain_bridge: true` and 3DPW as a validation domain.
5. **Evaluation**: add a per-domain MPJPE breakdown to `experiments/eval_variable_views.py` so the bridge benefit on 3DPW and MPI can be measured.

## 7. Relation to existing modules

- **v48 domain generalization**: v52 is complementary. v48 adapts batch statistics via FiLM/GRL at the feature level; v52 builds an explicit cross-dataset latent bridge with a contrastive loss and geometry-preserving gate.
- **v51 CDSVR**: v52 consumes the same per-view features but targets domain-level representation alignment rather than view-level reliability.
- **v50 SEFH**: v50 is a training-only auxiliary head; v52 is a feature transformation that affects both training and inference.
