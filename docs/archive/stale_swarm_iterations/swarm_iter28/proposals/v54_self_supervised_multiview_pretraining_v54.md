# v54 Self-Supervised Multi-View Pretraining for Robust Multi-View Fusion

**Author:** MotionFlow-MultiView design swarm
**Date:** 2026-08-09
**Tracking:** v54

## 1. Motivation

The MotionFlow pipeline now ends with v45 adaptive geometry fusion → v46 sparse-view generalization → v47/v49 temporal aggregation → v50 self-evolution feedback → v51 cross-domain sparse-view reliability → v52 uncertainty-weighted triangulation (UWT) → v53 physical-space calibration (PSC). These blocks progressively refine the *supervised* pose estimate, but they do not explicitly exploit the vast amount of unlabeled or lightly-labeled structure inherent in multi-view video itself.

v54 introduces a **self-supervised multi-view pretraining (SSMVP)** refinement head. Rather than adding new data loaders, it constructs auxiliary pretext tasks from the existing inputs: calibrated cameras, 2D detections, per-view features, and the current 3D pose estimate. These tasks regularize the fusion process, improve generalization under view dropout, and provide an identity-at-init bypass so that a trained v53 checkpoint remains unchanged when v54 is first enabled.

## 2. Architecture

v54 is implemented as a small post-PSC module `SelfSupervisedMultiViewPretrainingV54` placed immediately after `PhysicalSpaceCalibrationV53` in `OmniMultiViewFusionV5`.

### 2.1 Warm-startable 3D residual refine

Inputs:

- `pred_3d` (v53 output): `(B, T, J, 3)`
- `features`: `(B, T, V, J, d)`
- `points_2d`: `(B, T, V, J, 2)`
- `K`: `(B, T, V, 3, 3)`, `R`: `(B, T, V, 3, 3)`, `t`: `(B, T, V, 3)`
- `uwt_weights`: `(B, T, V, J)` from v52
- `view_mask`: `(B, T, V)`

The head first predicts a residual 3D correction:

```
ΔX = MLP_v54(LayerNorm(pred_3d))      ∈ (B, T, J, 3)
g  = σ(w_g),   w_g initialized to -6.0  ⇒ g ≈ 0 at init
X_out = pred_3d + g · ΔX
```

At initialization `g ≈ 0`, so `X_out = pred_3d` and the module is strictly identity. This satisfies the warm-start requirement for staged training on top of v52/v53.

### 2.2 Self-supervised pretext losses

The module computes three auxiliary losses that require no ground-truth 3D labels.

**Masked View Triangulation (MVT).** During training, randomly sample a subset of views `V_in` (keeping at least `v54_ssmvp_mvt_min_views`) and triangulate a 3D pose `X^masked` from only those views using the existing v52 UWT path. Reproject `X^masked` into the masked-out views and compare to the original 2D detections:

```
L_MVT = (1 / |V_mask| J) Σ_{v∈V_mask,j} ||Π_v(X^masked_j) - p_{v,j}||_2
```

This loss directly refines the same triangulation machinery used at test time and improves robustness to variable/sparse views.

**Cross-View Feature Consistency (CVFC).** For each joint, build a per-view feature prototype weighted by the v52 UWT weights and encourage view-invariant feature representations:

```
c_j = Σ_v w_{v,j} f_{v,j} / Σ_v w_{v,j}
L_CVFC = -Σ_{v,j} w_{v,j} log[ exp(c_j^T f_{v,j} / τ) / Σ_v' exp(c_j^T f_{v',j} / τ) ]
```

This loss forces the per-view encodings to agree on the same 3D joint semantics, which indirectly improves fusion quality.

**Temporal Pose Continuity (TPC).** Using the gated output `X_out`, predict the frame-to-frame residual with a 1D causal Conv1D over time and enforce smooth, physically plausible motion:

```
L_TPC = (1/(T-1)) Σ_{t=1}^{T-1} || (X_out_{t+1} - X_out_t) - Δ_θ(X_out_t) ||_2^2
```

### 2.3 Total v54 objective

```
L_v54 = λ_MVT   · L_MVT
      + λ_CVFC  · L_CVFC
      + λ_TPC   · L_TPC
```

`L_v54` is added to the existing geometry loss in `OmniMultiViewFusionV5.compute_losses` and is only active after `v54_ssmvp_warmup_epochs`.

## 3. Inputs and Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | v53 PSC output pose |
| `features` | `(B, T, V, J, d)` | Per-view per-joint features |
| `points_2d` | `(B, T, V, J, 2)` | Input 2D keypoints |
| `K, R, t` | `(B, T, V, 3, 3)` / `(B, T, V, 3)` | Calibrated cameras |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view/joint reliability weights |
| `view_mask` | `(B, T, V)` | Valid view mask |
| `X_out` | `(B, T, J, 3)` | Refined 3D pose (identity at init) |
| `ssmvp_loss` | scalar | Total v54 auxiliary loss |

## 4. Config Flags

```yaml
use_v54_self_supervised_multiview_pretraining: false
v54_ssmvp_hidden: 64
v54_ssmvp_n_layers: 2
v54_ssmvp_residual_gate_init: -6.0        # identity at init
v54_ssmvp_mvt_mask_prob: 0.25             # fraction of views to mask
v54_ssmvp_mvt_min_views: 2                # minimum views kept for MVT
v54_ssmvp_mvt_loss_weight: 0.1
v54_ssmvp_cvfc_loss_weight: 0.05
v54_ssmvp_tc_loss_weight: 0.01
v54_ssmvp_temperature: 0.1                # CVFC temperature τ
v54_ssmvp_warmup_epochs: 0                # epochs before loss is active
```

## 5. Expected MPJPE Impact

- **Sparse-view inference (2–3 views):** 3–6% relative reduction in `MPJPE@k`, because MVT explicitly trains the model to reconstruct missing views.
- **Full-view inference:** 0.5–1.5 mm absolute improvement via CVFC regularization and smoother temporal outputs from TPC.
- **Cross-domain transfer:** CVFC reduces reliance on dataset-specific view embeddings, which should lower the 3DPW actual-mode gap observed in v48.

## 6. Risks

1. **Compute overhead.** MVT requires an extra triangulation per training sample. We mitigate by masking only 25% of views and caching the masked triangulation when possible.
2. **Loss imbalance.** Self-supervised terms may dominate or vanish. We mitigate with warmup and per-loss gradient-norm clipping.
3. **Dependency on v52/v53 order.** v54 must run after PSC so that `pred_3d` is already physically calibrated. Changing the module order would break identity-at-init guarantees.

## 7. Implementation Plan

1. Create `motionflow_mv/fusion/self_supervised_multiview_pretraining_v54.py` with `SelfSupervisedMultiViewPretrainingV54` class, identity bypass, and the three loss functions.
2. Wire the module into `OmniMultiViewFusionV5.__init__` and `forward` after `PhysicalSpaceCalibrationV53`; store the auxiliary loss in `self._v54_ssmvp_loss`.
3. Add the config flags listed above to `OmniMultiViewFusionV5.__init__` and propagate them through the YAML config templates.
4. Add loss accumulation in `compute_losses` with warmup logic and an identity-init smoke test that confirms loading a v53 checkpoint with v54 enabled does not change `val_MPJPE` by more than 0.1 mm.
5. Run a smoke config (`configs/benchmark_v54_ssmvp_smoke.yaml`) on the local RTX 4090; if stable, append the entry to `scripts/launch_v33_a800_queue.py` and update `AGENTS.md`.
