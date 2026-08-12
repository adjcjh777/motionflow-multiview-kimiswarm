# v52: Bone-Length-Aware Fusion for Physical-Space Alignment

**Author:** design-swarm agent  
**Module name:** `bone_length_aware_fusion_v52`  
**Status:** Proposal (design-only)  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v45 adaptive geometry fusion, v46 sparse-view generalization, v50/v51 self-evolution feedback, v28/v40 physical-space alignment

## 1. Motivation

Current fusion modules (v25/v45, v46/v51, v47/v49-Lite) improve per-joint accuracy but treat joints largely independently during triangulation. **Bone-length constraints** are a strong, domain-invariant anthropometric prior: bone lengths are fixed within a clip and vary predictably across subjects. v52 injects this prior into the fusion stage, sitting between **multi-view fusion and calibration** and **physical-space alignment**. The module predicts per-bone length corrections and a kinematic-chain pose offset, then applies them to the triangulated 3-D pose. It is **warm-startable / identity at init** — final pose offsets are zero-initialized, so enabling it on a trained checkpoint leaves the baseline unchanged.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

The block is inserted **after** per-frame DLT triangulation (and after v45/v46/v50 reliability reweighting) and **before** temporal (v47/v49-Lite) and physical (v28/v40) heads.

### 2.2 Inputs and outputs

```
Inputs                         Outputs
pred_3d    : (B, T, J, 3)  ->  pred_3d_refined : (B, T, J, 3)
weights    : (B, T, V, J)      bone_loss       : scalar
visibility : (B, T, V, J)
view_mask  : (B, T, V)
parents    : (J,)
```

### 2.3 Bone extraction and encoding

For each frame, compute bone vectors, lengths, and uncertainty from the current pose and the per-joint precision matrices `L` already in scope:

```
b_j(t) = pred_3d[:,t,j] - pred_3d[:,t,parent(j)]   # (B, 3)
l_j(t) = || b_j(t) ||_2                              # (B,)
e_j    = MLP_bone( [ log(l_j + ε); log σ_l_j; d_j ] )  # (B, d_bone)
```

A bone is valid only if both endpoints are visible in at least `v52_bone_min_visible_views` views. `d_j` is a learned bone-type embedding.

### 2.4 Bone-length consistency network

A lightweight transformer processes encoded bones, and two heads predict additive length corrections and per-bone gates:

```
h_bone  = TransformerBoneEncoder(e)          # (B, num_bones, d_bone)
Δl      = MLP_length(h_bone)                 # zero-init
γ_bone  = Sigmoid(MLP_gate(h_bone))           # near-zero init
l'_j    = l_j + γ_bone_j * Δl_j
```

A single-layer skeleton GNN propagates corrected lengths back to per-joint 3-D offsets:

```
Δp_j     = GNN_skeleton( {Δl_k}, parents )  # zero-init output
pred_3d' = pred_3d + α * Δp_j
```

`α = v52_bone_residual_gate` defaults to `0.0`, making the block exactly identity at init.

### 2.5 Auxiliary bone-length loss

A learnable mean `μ_j` and variance `σ_j^2` per bone define the prior:

```
L_bone = - Σ_j w_j * [ 0.5 * log(2π σ_j^2) + 0.5 * (l'_j - μ_j)^2 / σ_j^2 ]
```

`w_j` down-weights low-visibility bones. The loss is active only when `v52_bone_loss_weight > 0`.

## 3. Configuration flags

```python
use_bone_length_aware_fusion_v52: bool = False
v52_bone_hidden: int = 64
v52_bone_n_layers: int = 2
v52_bone_n_heads: int = 4
v52_bone_dropout: float = 0.1
v52_bone_loss_weight: float = 0.01
v52_bone_residual_gate: float = 0.0
v52_bone_identity_init: bool = True
v52_bone_min_visible_views: int = 2
v52_bone_use_uncertainty: bool = True
```

## 4. Expected MPJPE impact

* **Sparse / cross-domain scenarios (v46, v51):** expected improvement **1.5–3.5 mm** on `MPJPE@2/3`.
* **Studio datasets (H36M):** identity-at-init means no regression; expected improvement **0.3–1.0 mm**.
* **WebBridge / 3DPW actual mode:** expected improvement **1–2 mm**.

## 5. Risks and mitigations

See `docs/swarm_iter26/reports/agent_bone_length_aware_fusion_risks.md` for the full register. Top risks include warm-start failure, canonical-skeleton collapse, conflict with v28/v40, cross-dataset skeleton mismatch, and sparse-view degradation.

## 6. 5-step implementation plan

1. **Prototype the standalone module.** Create `motionflow_mv/fusion/bone_length_aware_fusion_v52.py` with bone extraction, transformer, length/gate heads, and skeleton GNN. Add unit tests for identity-at-init and shape correctness.

2. **Wire into `OmniMultiViewFusionV5`.** Add the v52 flags to the constructor. Call the block after DLT triangulation, passing `pred_3d`, `weights`, `visibility`, and the runtime parent list. Add the auxiliary `bone_loss` to `epi_loss` with weight `v52_bone_loss_weight`.

3. **Add warm-start smoke tests.** With `v52_bone_residual_gate=0.0`, assert the output pose equals the input pose and gradients flow. Gradually increase the gate and verify the bone-length loss decreases.

4. **Run smoke training.** Use `configs/benchmark_v52_bone_length_aware_fusion_smoke.yaml` with `v52_bone_loss_weight=0.01`. Compare against the v51 baseline on a small mixed manifest. Target: finite loss, no NaNs, val_MPJPE within 5 mm of baseline after 1 epoch.

5. **Scale to full A800 run.** If smoke passes, add an A800 queue entry in `scripts/launch_v33_a800_queue.py` on top of the strongest v46/v51 checkpoint. Report epoch-1 `MPJPE@k`, per-domain metrics, and learned bone-length prior statistics.

## 7. Paper story fit

v52 reinforces the paper’s optimized motionflow pipeline by injecting a lightweight **physical skeleton prior** directly into the multi-view fusion stage, bridging geometry-based triangulation and downstream physical-space alignment.
