# v55 Outlier-Robust Reliability (OR2)

**Tracking issue:** #208  
**Base branch:** `v55-orr`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

## 1. Module name and one-line purpose

**Module:** `OutlierRobustReliabilityV55` → `motionflow_mv/fusion/outlier_robust_reliability_v55.py`

**One-line purpose:** Refine the per-view joint reliability weights produced by v45 adaptive geometry fusion with a learned Cauchy M-estimator before v52 uncertainty-weighted triangulation, so that gross outlier views are down-weighted while preserving the warm-started baseline at initialization.

## 2. Placement in `OmniMultiViewFusionV5` forward pass

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v55 OutlierRobustReliabilityV55
    (consumes pred_3d_init, weights_init, points_2d, K, R, t, view_mask)
    → weights_orr, orr_loss
    ↓
v52 UncertaintyWeightedTriangulationV52
    (consumes pred_3d_init, weights_orr)
    → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

OR2 is placed **immediately after v45 geometry fusion and before v52 UWT**. It does not replace the v52 weight-learning branch; it supplies v52 with a pre-conditioned, outlier-robust weight map so that the physical-space calibration stages downstream receive a cleaner triangulated signal.

## 3. Inputs, outputs, and shapes

Let `B` be batch size, `T` clip length, `V` number of views, `J` number of joints, and `D` the model dimension (default 64).

**Inputs**

| Symbol | Tensor shape | Description |
|--------|--------------|-------------|
| `pred_3d_init` | `(B, T, J, 3)` | Initial triangulated 3D pose from v45. |
| `weights_init` | `(B, T, V, J)` or `(B, T, V, J, 1)` | Raw per-view joint weights from v45 geometry fusion. |
| `points_2d` | `(B, T, V, J, 2)` | Input 2D keypoints. |
| `confidences` | `(B, T, V, J)` | Detector confidence scores. |
| `K`, `R`, `t` | `(B, T, V, 3, 3)`, `(B, T, V, 3, 3)`, `(B, T, V, 3)` | Camera intrinsics and extrinsics. |
| `view_mask` | `(B, T, V)` or `(B, T, V, 1)` | Binary/dropout mask from v46 sparse-view augmentation. |

**Outputs**

| Symbol | Tensor shape | Description |
|--------|--------------|-------------|
| `weights_orr` | `(B, T, V, J)` | Outlier-robust reliability weights, same shape as `weights_init`, clamped to `[v55_orr_min_weight, 1.0]`. |
| `orr_loss` | scalar | Optional auxiliary inlier-consistency loss. |

**Shapes preserved:** `weights_orr` broadcasts directly into the existing v52 UWT path with no changes to downstream tensor shapes.

## 4. Architecture

### 4.1 Feature construction

For each `(view, joint)` token, concatenate:

1. **Geometry bias:** ray direction, reprojection residual of `pred_3d_init` into the view, epipolar distance to the nearest other view, and the triangulation angle.
2. **Feature bias:** the v45 learned view-joint feature (or pooled ST-transformer token).
3. **Input weight signal:** `weights_init` and `confidences`.

The concatenated vector is fed through a -layer per-token MLP (`v55_orr_hidden`, ReLU, LayerNorm) to produce a per-token outlier score `s_vj`.

### 4.2 Cauchy M-estimator

Given the outlier score `s_vj`, compute the inlier likelihood under a Cauchy kernel:

```text
p_inlier(v,j) = 1 / (1 + (s_vj / γ)^2)
```

where `γ = softplus(v55_orr_cauchy_gamma) + ε` is initialized so that `γ ≈ 1.0` at start. The refined weight is:

```text
weights_orr_raw = weights_init * p_inlier
```

### 4.3 Residual gate (identity-at-init)

A learned scalar gate `α` is initialized to `sigmoid(-6.0) ≈ 0.0025` and applied as:

```text
weights_orr = α * weights_orr_raw + (1 - α) * weights_init
```

At initialization, `weights_orr ≈ weights_init`, so a v54-v2 checkpoint loaded with v55 enabled produces the same triangulation as before.

### 4.4 Auxiliary loss

The module contributes a small auxiliary loss `orr_loss` only when `v55_orr_loss_weight > 0` and the warmup epoch is reached:

- **Inlier reprojection loss:** weighted mean reprojection error of the inlier-selected subset, encouraging `p_inlier` to select views whose reprojections agree with the refined pose.
- **Entropy regularizer (optional):** a tiny term that prevents the inlier distribution from collapsing to all-zero or all-one.

### 4.5 Config flags and defaults

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v55_outlier_robust_reliability` | bool | `False` | Master toggle. |
| `v55_orr_hidden` | int | `64` | MLP hidden dimension. |
| `v55_orr_n_layers` | int | `2` | MLP depth. |
| `v55_orr_identity_init` | bool | `True` | Zero-initialize final output layer and gate. |
| `v55_orr_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init. |
| `v55_orr_cauchy_gamma_init` | float | `1.0` | Initial Cauchy scale `γ`. |
| `v55_orr_use_geometry_bias` | bool | `True` | Include reprojection/ray/epipolar features. |
| `v55_orr_use_feature_bias` | bool | `True` | Include learned v45 feature tokens. |
| `v55_orr_min_weight` | float | `0.05` | Floor for refined weights. |
| `v55_orr_loss_weight` | float | `0.01` | Multiplier on `orr_loss`. |
| `v55_orr_warmup_epochs` | int | `0` | Epochs before `orr_loss` contributes. |
| `v55_orr_use_entropy_reg` | bool | `False` | Toggle entropy regularizer. |

## 5. Expected MPJPE impact and main risks

### Expected impact

| Scenario | Expected change |
|----------|---------------|
| Smoke `val_MPJPE@full` | `-0.5` to `-1.5` mm |
| Full run `val_MPJPE@full` | `-0.8` to `-2.0` mm |
| `MPJPE@2` / `MPJPE@3` sparse views | `-2` to `-4` mm |
| 3DPW actual / cross-domain | `-1` to `-3` mm (outlier-heavy captures) |

The largest gains are expected in sparse-view regimes where a single bad view can dominate the triangulation, and on outlier-heavy sequences where v45 weights are noisy.

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **Gate fails closed or open** | `weights_orr` collapses to uniform or to zero. | Initialize gate to `-6.0`; clamp `weights_orr` to `[min_weight, 1.0]`; unit-test identity-at-init. |
| **Cauchy scale drifts** | `γ` becomes very small and rejects valid views. | Parameterize `γ` via `softplus + 0.5` lower bound; add loss weight warmup. |
| **v52 UWT overwrites OR2 weights** | No visible gain if v52 re-learns its own weights. | Feed `weights_orr` as the *initial* v52 weight and make v52 residual small; ablate `v55_orr_loss_weight`. |
| **Conflicts with v51 CDSVR reliability** | Double suppression of rare but correct views. | Gate OR2 on the v51 domain-conditioned reliability only at init; keep both branches additive. |
| **Sparse-view degeneracy** | With `min_views=2`, OR2 may reject a needed second view. | Enforce `v55_orr_min_weight > 0`; mask the top-`min_views` highest `weights_init` from being suppressed. |

## 6. Smoke acceptance criteria

Run `bash scripts/run_v55_orr_smoke_local_4090.sh` on the local RTX 4090, warm-started from the best v54-v2 checkpoint.

- **Identity-at-init:** loading the v54-v2 checkpoint with v55 enabled and no training step changes `val_MPJPE@full` by `< 0.1` mm.
- **Baseline proximity:** `val_MPJPE@full` is within `1 mm` of the v54-v2 baseline after the first epoch.
- **Sparse-view improvement:** `MPJPE@2` and `MPJPE@3` are not worse than the v54-v2 baseline; target `≥ 1 mm` improvement.
- **Stability:** no NaN, Inf, or OOM through at least one full epoch.
- **Weight sanity:** at least `95%` of `weights_orr` are finite and in `[0.05, 1.0]`; mean refined weight is between `0.3` and `0.9` for non-masked tokens.
- **Outlier rejection:** on synthetic outlier injection (add one noisy view), the corrupted view receives a weight below `0.2` in `≥ 80%` of cases.

If all criteria pass, add/confirm the A800 queue entry and run the full v55-orr variant on top of v54-v2.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/outlier_robust_reliability_v55.py` — `OutlierRobustReliabilityV55` module.
- `configs/benchmark_v55_orr_smoke.yaml` — smoke config copied from v54-v2 smoke with v55 flags enabled.
- `scripts/run_v55_orr_smoke_local_4090.sh` — smoke launch script that warm-starts from the best v54-v2 checkpoint.
- `tests/test_outlier_robust_reliability_v55.py` — unit tests for identity-at-init, Cauchy scale bounds, weight clamping, and outlier rejection.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add constructor flag `use_v55_outlier_robust_reliability` and instantiate `OutlierRobustReliabilityV55` when enabled.
  - Insert the OR2 call **between** v45 geometry fusion and v52 UWT, passing `weights_orr` into v52 instead of `weights_init`.
  - Add `orr_loss` to the `epi_loss` dictionary with key `v55_orr`.

- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Aggregate `loss_dict["v55_orr"]` (if present) with `v55_orr_loss_weight` only after `v55_orr_warmup_epochs`.

- `scripts/launch_v33_a800_queue.py`
  - Add entry `v55_outlier_robust_reliability_on_v54` to queue the full v55 variant on top of the best v54-v2 checkpoint.

- `AGENTS.md`
  - Add a new "v55 outlier-robust reliability conventions" section mirroring the v52/v53/v54 convention blocks, once the design is approved.
