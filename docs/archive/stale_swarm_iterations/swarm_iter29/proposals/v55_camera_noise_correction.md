# v55 Camera Noise Correction (CNC)

## 1. Module name and one-line purpose

- **Module:** `CameraNoiseCorrectionV55`
- **File:** `motionflow_mv/fusion/camera_noise_correction_v55.py`
- **One-line purpose:** A lightweight, identity-at-init correction head that refines raw 2-D keypoints and camera intrinsics/extrinsics before v52 UWT, guided by the v25/v45 initial pose to absorb detector jitter and small camera drift without disturbing the established v54 PSC-v2 pipeline.

## 2. Where it sits in the `OmniMultiViewFusionV5` forward pass

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v55 CameraNoiseCorrectionV55
    (consumes points_2d, confidences, K, R, t, pred_3d_init, view_mask)
    → points_2d_corr, K_corr, R_corr, t_corr, cnc_loss
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

v55 CNC is placed **after** the initial v25/v45 triangulation (so it has a rough 3-D prior) and **before** v52 UWT (so the rest of the stack receives cleaner observations and cameras). It does not replace any existing block; it is an input-space refinement layer that feeds all downstream triangulation and calibration modules.

## 3. Inputs, outputs, and shapes

**Inputs**

| Symbol | Shape | Description |
|---|---|---|
| `points_2d` | `(B, T, V, J, 2)` | Raw 2-D keypoints from the detector. |
| `confidences` | `(B, T, V, J)` | Per-keypoint detection confidence. |
| `K` | `(B, T, V, 3, 3)` | Camera intrinsics. |
| `R` | `(B, T, V, 3, 3)` | Camera rotations. |
| `t` | `(B, T, V, 3)` | Camera translations. |
| `pred_3d_init` | `(B, T, J, 3)` | Initial 3-D estimate from v25/v45. |
| `view_mask` | `(B, T, V)` | Valid view mask. |

**Outputs**

| Symbol | Shape | Description |
|---|---|---|
| `points_2d_corr` | `(B, T, V, J, 2)` | Corrected 2-D keypoints. |
| `K_corr` | `(B, T, V, 3, 3)` | Corrected intrinsics. |
| `R_corr` | `(B, T, V, 3, 3)` | Corrected rotations. |
| `t_corr` | `(B, T, V, 3)` | Corrected translations. |
| `cnc_loss` | `scalar` | Auxiliary reprojection-consistency and correction-regularization loss. |

## 4. Architecture

### 4.1 Per-(view, joint) feature vector

For each view `v` and joint `j`, build a feature from:

- the v25/v45 feature token `f_{v,j} ∈ R^d` (if available; otherwise a learned placeholder);
- reprojection residual `r_{v,j} = ||p_{v,j} − Π(K_v, R_v, t_v, X_j)||_2`;
- log-residual `log(r_{v,j} + ε)`;
- ray direction through the 2-D point in world space;
- camera center `c_v = −R_v^T t_v`;
- detection confidence `conf_{v,j}`.

### 4.2 2-D keypoint correction head

A two-layer MLP with hidden dimension `v55_cnc_hidden` outputs a bounded 2-D offset:

```
Δp_{v,j} = tanh(MLP_2d(f_{v,j}, r_{v,j}, log r_{v,j}, ray_{v,j}, c_v, conf_{v,j})) · v55_cnc_max_2d_offset_px
```

The final layer of `MLP_2d` is zero-initialized so `Δp = 0` at init.

### 4.3 Camera correction head

Per-view features are pooled over joints (confidence-weighted mean) and passed to a second two-layer MLP that predicts:

| Parameter | Output | Bound |
|---|---|---|
| focal scale | `Δf_v` | `[-v55_cnc_max_focal_scale, v55_cnc_max_focal_scale]` |
| principal-point offset | `Δo_v ∈ R^2` | `[-v55_cnc_max_pp_offset_px, v55_cnc_max_pp_offset_px]` |
| rotation axis-angle | `Δω_v ∈ R^3` | magnitude `≤ v55_cnc_max_rot_deg` |
| translation offset | `Δt_v ∈ R^3` | `[-v55_cnc_max_t_mm, v55_cnc_max_t_mm]` |

All final output layers are zero-initialized.

### 4.4 Corrected quantities

```
K_corr[v] = K[v] · diag(1 + Δf_v, 1 + Δf_v, 1),   with principal point += Δo_v
R_corr[v] = exp([Δω_v]_×) · R[v]
t_corr[v] = t[v] + Δt_v
points_2d_corr[v,j] = points_2d[v,j] + Δp_{v,j}
```

### 4.5 Residual gate for identity-at-init

A scalar gate is applied to all corrections:

```python
gate = torch.sigmoid(v55_cnc_residual_gate_init)  # default -6.0 → ≈ 0.0025
points_2d_corr = points_2d + gate * delta_p
K_corr         = K + gate * delta_K
R_corr         = compose_small_rotation(gate * delta_omega, R)
t_corr         = t + gate * delta_t
```

Because `gate ≈ 0` and the final MLP layers are zero-initialized, `K_corr = K`, `R_corr = R`, `t_corr = t`, and `points_2d_corr = points_2d` at initialization. A warm-started v54 checkpoint therefore loads unchanged.

### 4.6 Losses

| Loss | Weight flag | Description |
|---|---|---|
| `L_reproj` | `v55_cnc_loss_weight` | Reprojection consistency of `pred_3d_init` against `points_2d_corr` under `K_corr, R_corr, t_corr`, weighted by `confidences`. |
| `L_reg` | `v55_cnc_reg_weight` | L2 magnitude of all corrections, encouraging small corrections unless the data demands them. |
| `L_temporal` | `v55_cnc_temporal_weight` (optional) | Temporal smoothness of `Δp` and camera corrections across frames. |

Total auxiliary loss: `L_cnc = v55_cnc_loss_weight * L_reproj + v55_cnc_reg_weight * L_reg + v55_cnc_temporal_weight * L_temporal`.

A warmup guard `v55_cnc_warmup_epochs` prevents `L_cnc` from contributing to the total loss until the specified number of epochs have elapsed.

## 5. Expected MPJPE impact (full/sparse views) and main risks

| View setting | Expected MPJPE impact |
|---|---|
| Full views | `−0.2 to −0.8 mm` |
| Sparse views (`@2/3`) | `−1.0 to −2.5 mm` |
| Noisy or calibration-shifted data | up to `−2.5 mm` |

**Main risks**

- **Over-correction and 2-D drift:** Large learned offsets can distort the true keypoint locations and raise MPJPE.
  - *Mitigation:* Bound `Δp` and camera corrections; apply the `−6.0` residual gate; regularize correction magnitude; clamp offsets during warmup.
- **Camera parameter instability:** Correcting extrinsics/intrinsics per-frame can introduce non-physical jumps.
  - *Mitigation:* Bound all camera corrections tightly; add temporal-smoothness loss; make camera correction optional via `v55_cnc_correct_intrinsics` / `v55_cnc_correct_extrinsics`.
- **Double-counting with v52 uncertainty:** v52 already down-weights noisy views; v55 should clean the inputs rather than reweight them.
  - *Mitigation:* Keep the module strictly in input space; do not output view weights; use only reprojection consistency and correction regularization as losses.
- **Identity-at-init failure:** A v54 checkpoint could regress if the gate or final layers are not zero-initialized.
  - *Mitigation:* Zero-initialize all final MLP projections; set gate logit `−6.0`; add unit test `||points_2d_corr − points_2d||_∞ < 1e-4` and `||K_corr − K||_F < 1e-4` at init.

## 6. Smoke acceptance criteria

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Correction bounds: at least `95%` of 2-D corrections satisfy `||Δp||_∞ ≤ v55_cnc_max_2d_offset_px`; camera corrections stay within their declared bounds.
- Reprojection sanity: mean reprojection error of `pred_3d_init` under the corrected cameras/keypoints is `≤` the baseline reprojection error.
- `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.

## 7. Required new files and files to modify

**New files**

- `motionflow_mv/fusion/camera_noise_correction_v55.py` — `CameraNoiseCorrectionV55` module.
- `configs/benchmark_v55_camera_noise_correction_smoke.yaml` — smoke config copied from the v54 smoke with v55 flags enabled.
- `scripts/run_v55_camera_noise_correction_smoke_local_4090.sh` — smoke launch script warm-starting from the best v54 checkpoint.
- `tests/test_camera_noise_correction_v55.py` — unit tests for identity-at-init, bounded corrections, camera composition, and gradient flow.

**Files to modify**

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add constructor flags under the v55 block.
  - Instantiate `CameraNoiseCorrectionV55` when `use_v55_camera_noise_correction=True`.
  - Insert the module call immediately after the v25/v45 geometry fusion block, before v52 UWT.
  - Pass corrected `points_2d_corr`, `K_corr`, `R_corr`, `t_corr` to v52/v53/v54.
  - Add `cnc_loss` to the `epi_loss` dictionary.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Aggregate `loss_dict["v55_cnc"]` into the total loss with weight `v55_cnc_loss_weight` and warmup guard `v55_cnc_warmup_epochs`.
- `scripts/launch_v33_a800_queue.py`
  - Add an A800 full-run entry `v55_camera_noise_correction_on_v54` warm-starting from the best v54 checkpoint.
