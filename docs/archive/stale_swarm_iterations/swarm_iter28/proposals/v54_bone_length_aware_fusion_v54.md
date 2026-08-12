# v54 Bone-Length-Aware Fusion (BLAF)

## 1. Motivation

The current MotionFlow-MultiView pipeline triangulates per-joint 3D positions (v25/v45), reweights views by uncertainty (v52), and applies physical-space calibration (v53). However, joints are still refined largely independently. Human skeletons have strong bone-length constraints: the lengths of the humerus, femur, tibia, etc., are nearly constant for a given subject across a sequence, and their ratios follow a low-dimensional anthropometric manifold. Violating these constraints produces anatomically implausible poses and amplifies MPJPE, especially under sparse views where per-joint triangulation is noisy.

v54 introduces a **Bone-Length-Aware Fusion (BLAF)** module that sits on top of v52/v53, extracts a canonical bone-length profile from the current pose and multi-view features, and uses it to gate a per-joint residual correction. The module is warm-startable/identity-at-init so that enabling it does not degrade the v53 baseline before it has learned anything.

## 2. Architecture

BLAF is placed **after** `PhysicalSpaceCalibrationV53` and **before** the final residual MLP in `OmniMultiViewFusionV5`. It receives:

- `pred_3d`: `(B, T, J, 3)` — triangulated 3D pose after v53.
- `features`: `(B, T, J, d)` — per-joint geometry-fusion tokens from v25/v45.
- `uwt_weights`: `(B, T, V, J)` or `(B, T, J)` — per-joint confidence from v52 (optional).
- `view_mask`: `(B, T, V)` — visibility mask (optional, for sparse-view safety).

Output:

- `refined_3d`: `(B, T, J, 3)` — bone-length-aware 3D pose.
- `blaf_loss`: scalar auxiliary loss for bone-length consistency.

Internally, BLAF has three sub-components:

1. **Bone-length encoder** — computes current bone lengths and directions from the parent-child skeleton graph. For each bone `b = (parent, child)`:
   
   \[
   \mathbf{u}_b = p_{child} - p_{parent}, \quad
   \ell_b = \|\mathbf{u}_b\|_2
   \]

2. **Canonical bone-length estimator** — a lightweight MLP that maps pooled per-bone features (concatenated parent/child features plus optional UWT confidence) to a canonical length offset:
   
   \[
   \Delta \ell_b^{*} = \mathrm{MLP}_{\mathrm{canon}}(\mathrm{cat}[f_{parent}, f_{child}, \bar{w}_b]) \in \mathbb{R}
   \]
   
   where `\bar{w}_b` is the average UWT weight over views for the two joints. The final canonical target is `\ell_b^{*} = \ell_b + \Delta \ell_b^{*}`.

3. **Residual gate** — predicts a per-joint residual gate `g_j \in [0, 1]` that blends the original v53 pose with a bone-length-corrected pose:
   
   \[
   \hat{p}_{child}^{(b)} = p_{parent} + \frac{\ell_b^{*}}{\ell_b} \mathbf{u}_b
   \]
   
   \[
   \Delta p_j = \mathrm{MLP}_{\mathrm{res}}(\mathrm{cat}[f_j, \ell_{b(j)}^{*}, \ell_{b(j)}]) \in \mathbb{R}^3
   \]
   
   \[
   p_j^{\text{out}} = p_j + g_j \cdot \Delta p_j
   \]

   The gate logits are initialized so that `g_j \approx 0` at initialization (identity-at-init). Specifically, the final linear layer of the gate MLP is initialized to zero and a bias `v54_blaf_residual_gate_init = -6.0` is added, giving `g_j \approx 0.0025`.

## 3. Inputs / Outputs (tensor shapes)

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Input 3D pose from v53 |
| `features` | `(B, T, J, d)` | Per-joint geometry tokens from v25/v45 |
| `uwt_weights` | `(B, T, V, J)` or `(B, T, J)` | Optional v52 uncertainty/confidence |
| `bone_mask` | `(B, T, B_bones)` | Mask for missing joints/parents |
| **Output** `refined_3d` | `(B, T, J, 3)` | Bone-length-aware pose |
| **Output** `blaf_loss` | scalar | Auxiliary bone-length consistency loss |

## 4. Config flags

```
use_v54_bone_length_aware_fusion: bool = False
v54_blaf_hidden: int = 64
v54_blaf_n_layers: int = 2
v54_blaf_identity_init: bool = True
v54_blaf_residual_gate_init: float = -6.0
v54_blaf_use_uwt_weights: bool = True
v54_blaf_canonical_mode: str = "sequence"   # "sequence" | "subject" | "global"
v54_blaf_loss_weight: float = 0.01
v54_blaf_bone_loss_weight: float = 0.1
v54_blaf_warmup_epochs: int = 0
v54_blaf_min_bone_length: float = 1e-3
```

- `v54_blaf_canonical_mode="sequence"` learns a per-sequence canonical profile (via EMA over the batch). `"subject"` would require a subject ID and is reserved for future data-loader work. `"global"` uses a single learned vector shared across all data.
- `v54_blaf_bone_loss_weight` scales the auxiliary loss that encourages temporal and view-consistent bone lengths.

## 5. Expected MPJPE impact

- **Full-view inference:** modest gain of ~0.3–0.8 mm by suppressing anatomically inconsistent residuals.
- **Sparse/variable-view inference (`MPJPE@2`, `MPJPE@3`):** larger gain of 1.5–3.0 mm, because bone-length constraints regularize noisy per-view triangulation.
- **Warm-start verification:** loading a v53 checkpoint with v54 enabled should change `val_MPJPE@full` by ≤ 0.1 mm before training.

## 6. Risks

1. **Conflict with v53 bone-scale loss** — both modules reason about bone lengths. Mitigation: make BLAF identity-at-init and keep its loss weight small initially; gate it with `v54_blaf_warmup_epochs`.
2. **Overfitting to canonical lengths** — a fixed bone-length profile may hurt extreme poses. Mitigation: learn an *offset* relative to the current length, not an absolute replacement, and use the residual gate.
3. **Extra compute** — bone MLPs add parameters. Mitigation: keep hidden dim small (`d=64` default, 2 layers) and share MLP weights across symmetric bones.
4. **Sparse-view instability** — missing parents/children break bone computation. Mitigation: use `bone_mask` and fall back to identity when fewer than 2 joints of a bone are visible.

## 7. 5-step implementation plan

1. **Module:** create `motionflow_mv/fusion/bone_length_aware_fusion_v54.py` with `BoneLengthAwareFusionV54` implementing the encoder, canonical estimator, and residual gate.
2. **Wiring:** in `OmniMultiViewFusionV5.__init__` add the v54 flags and instantiate the module; in `forward` call it after v53 and before the final residual MLP.
3. **Loss:** register `v54_blaf_loss` in `forward` and add it to the total loss inside `get_loss` with warm-up gating.
4. **Smoke test:** run `configs/benchmark_v54_blaf_smoke.yaml` on RTX 4090; verify identity-at-init (Δ MPJPE ≤ 0.1 mm) and stable training for 1 epoch.
5. **Ablate:** compare `v53` vs `v53 + v54` on the v49/v50/v52/v53 baseline; report `MPJPE@full`, `MPJPE@2/3/4`, and per-bone-length consistency metrics.
