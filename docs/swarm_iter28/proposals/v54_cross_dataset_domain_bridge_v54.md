# v54 Cross-Dataset Domain Bridge (CDDB)

## 1. Motivation

MotionFlow-MultiView mixes Human3.6M, MPI-INF-3DHP, 3DPW, and WebBridge data. v48 adapts feature tokens and v51/v52/v53 refine fusion, triangulation, and physical-space calibration, yet the final 3D pose can still carry dataset-specific biases in scale, root height, and joint distribution. These biases limit cross-dataset generalization and the downstream optimized motionflow pipeline.

v54 introduces a **Cross-Dataset Domain Bridge (CDDB)** after `PhysicalSpaceCalibrationV53`. It learns a domain-conditional affine normalization into a shared canonical pose space, applies a skeleton-aware residual refinement, and maps back to the original domain. The module is warm-startable/identity-at-init so that enabling it preserves the v53 baseline before it learns.

## 2. Architecture

CDDB is placed **after** v53 and **before** the final residual MLP in `OmniMultiViewFusionV5`. Inputs:

- `pred_3d_psc`: `(B, T, J, 3)` — 3D pose from v53.
- `features`: `(B, T, J, d)` — per-joint geometry tokens from v25/v45.
- `domain_id`: `(B,)` or `(B, T)` — integer domain labels.
- `uwt_weights`: `(B, T, V, J)` or `(B, T, J)` — optional v52 confidence.

Outputs:

- `refined_3d`: `(B, T, J, 3)` — domain-bridged 3D pose.
- `cddb_loss`: scalar auxiliary canonical-consistency + residual loss.

**Domain-conditional affine normalization (DAN).** A small MLP maps a per-domain embedding `e_d = E[domain_id] ∈ R^h` to per-joint log-scale and offset:

\[
\log s_d,\, o_d = \mathrm{MLP}_{\mathrm{aff}}(e_d) \in \mathbb{R}^{J \times 3}
\]

The canonical pose is

\[
\tilde{p}_j = \exp(\log s_d) \odot (p_j - o_d)
\]

The final layer of `MLP_aff` is initialized to zero, so `s_d = 1` and `o_d = 0` (identity-at-init).

**Skeleton-aware canonical refiner (SCR).** A lightweight MLP predicts a gated residual on the canonical pose:

\[
\Delta p_j = \mathrm{MLP}_{\mathrm{ref}}(\mathrm{cat}[\tilde{p}_j, f_j, \bar{w}_j]) \in \mathbb{R}^3
\]

\[
\tilde{p}'_j = \tilde{p}_j + g_j \odot \Delta p_j, \quad g_j = \mathrm{sigmoid}(\mathrm{MLP}_{\mathrm{gate}}(\cdot) + b_{\mathrm{gate}})
\]

`b_gate = v54_cddb_residual_gate_init = -6.0` gives `g_j  0.0025` at start, preserving the v53 output.

**Domain-conditional denormalization.** The refined canonical pose is mapped back with the inverse affine transform:

\[
p'_j = \frac{\tilde{p}'_j}{\exp(\log s_d)} + o_d
\]

At init this is exactly identity.

**Canonical consistency loss.** A moment-matching term over the batch keeps the canonical space domain-agnostic:

\[
\mathcal{L}_{\mathrm{consist}} = \sum_d \|\mu_d - \bar{\mu}\|_2^2 + \|\sigma_d^2 - \bar{\sigma}^2\|_2^2
\]

where `μ_d, σ_d^2` are mean and variance of canonical poses for domain `d`, and `μ̄, σ^2` are global batch statistics.

## 3. Inputs / Outputs (tensor shapes)

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc` | `(B, T, J, 3)` | Input 3D pose from v53 |
| `features` | `(B, T, J, d)` | Per-joint geometry tokens from v25/v45 |
| `domain_id` | `(B,)` or `(B, T)` | Integer domain label per sample/frame |
| `uwt_weights` | `(B, T, V, J)` or `(B, T, J)` | Optional v52 confidence |
| **Output** `refined_3d` | `(B, T, J, 3)` | Domain-bridged 3D pose |
| **Output** `cddb_loss` | scalar | Auxiliary canonical-consistency + residual loss |

## 4. Config flags

```
use_v54_cross_dataset_domain_bridge: bool = False
v54_cddb_hidden: int = 64
v54_cddb_n_layers: int = 2
v54_cddb_num_domains: int = 6
v54_cddb_identity_init: bool = True
v54_cddb_residual_gate_init: float = -6.0
v54_cddb_use_uwt_weights: bool = True
v54_cddb_use_canonical_refiner: bool = True
v54_cddb_consistency_weight: float = 0.01
v54_cddb_residual_weight: float = 0.001
v54_cddb_loss_weight: float = 0.01
v54_cddb_warmup_epochs: int = 0
v54_cddb_unknown_domain_id: int = -1
```

`v54_cddb_unknown_domain_id` labels samples without a domain label; they receive a learned "unknown" embedding that defaults to identity behavior.

## 5. Expected MPJPE impact

- **In-domain val (H36M / MPI):** neutral to +0.2 mm.
- **Cross-dataset val (3DPW / WebBridge):** gain of ~1.0–2.5 mm by removing residual domain biases.
- **Sparse/variable views (`MPJPE@2/3`):** gain of ~0.8–2.0 mm from canonical priors.
- **Warm-start:** loading a v53 checkpoint with v54 enabled should change `val_MPJPE@full` by ≤ 0.1 mm before training.

## 6. Risks

1. **Affine scale/offset drift.** Mitigation: log-scale parameterization, zero final layer, clipping, and residual regularization.
2. **Refiner overfits to dominant domain.** Mitigation: consistency loss and per-domain validation; can disable refiner via flag.
3. **Conflict with v53.** Mitigation: identity-at-init, gated residual, loss warm-up.
4. **Missing domain labels.** Mitigation: `unknown_domain_id` embedding defaults to identity.

## 7. 5-step implementation plan

1. **Module:** create `motionflow_mv/fusion/cross_dataset_domain_bridge_v54.py` with `CrossDatasetDomainBridgeV54`.
2. **Wiring:** add flags to `OmniMultiViewFusionV5.__init__`; call the module after v53 and before the final residual MLP, passing `domain_id` and v52 UWT weights.
3. **Loss:** register `v54_cddb_loss` in `forward` and add it to `get_loss`, gated by `v54_cddb_warmup_epochs`.
4. **Smoke test:** create `configs/benchmark_v54_cddb_smoke.yaml` and run on RTX 4090; verify identity-at-init (Δ MPJPE ≤ 0.1 mm) and 1-epoch stability.
5. **Ablate:** compare `v53` vs `v53 + v54`; report `MPJPE@full`, `MPJPE@2/3/4`, and per-domain `MPJPE`.
