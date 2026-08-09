# v52 Camera Noise Correction (CNC)

## Motivation

MotionFlow currently treats calibrated intrinsics `K` and extrinsics `(R, t)` as constants. In reality, calibration always contains noise: reprojection residuals, imperfect checkerboard fits, temporal drift, and domain-specific systematic biases. v52 adds a learned **camera-noise correction head** that refines `K, R, t` from the 2D keypoint observations before triangulation, directly supporting the paper narrative

> multi-view video → human pose extraction → multi-view fusion **and calibration** → physical-space alignment → optimized MotionFlow.

The module is warm-startable/identity-at-init: at initialization it predicts zero correction, reproducing the existing v45/v46/v51 baseline exactly.

---

## Architecture

The head is inserted **after** feature extraction and **before** DLT triangulation and Gauss–Newton refinement. Shapes use `N = B·T`.

### 1. Inputs

* `points_2d` ∈ `R^(N,V,J,2)` — 2D joint detections.
* `confidences` ∈ `R^(N,V,J)` — detection confidence.
* `feat` ∈ `R^(N,V,J,d)` — per-view encoder features.
* `K` ∈ `R^(N,V,3,3)`, `R` ∈ `R^(N,V,3,3)`, `t` ∈ `R^(N,V,3)` — camera parameters.
* `view_mask` ∈ `{0,1}^(N,V)` — active views.

Per-view context is obtained by confidence-weighted pooling over joints:

```text
g_v = Σ_j confidences_(v,j) · feat_(v,j) / Σ_j confidences_(v,j)   ∈ R^(N,V,d)
```

### 2. Correction MLP

A shared MLP (`v52_cnc_num_layers` layers, `v52_cnc_hidden` width) outputs correction terms, all zero-initialized for identity-at-init:

```text
Δlog_f ∈ R^(N,V)       # relative focal-length perturbation
Δcx, Δcy ∈ R^(N,V)     # principal-point offsets (pixels)
ξ ∈ R^(N,V,3)          # so(3) rotation correction
δt ∈ R^(N,V,3)         # translation correction (meters)
```

The final layer is multiplied by a residual gate `sigmoid(v52_cnc_residual_gate_init)`. With default `-6.0`, the gate is `≈ 0.0025`, so the module is effectively a no-op at start and warm-starts safely from pretrained checkpoints.

### 3. Bounded camera updates

```text
K'_v = K_v · diag(1 + clamp(Δlog_f_v, -f_max, f_max),
                  1 + clamp(Δlog_f_v, -f_max, f_max), 1)
K'_v[..., 2] += clamp(Δcx_v, -pp_max, pp_max)
K'_v[..., 5] += clamp(Δcy_v, -pp_max, pp_max)

R'_v = exp([clamp(ξ_v, -θ_max, θ_max)]) · R_v
t'_v = t_v + clamp(δt_v, -t_max, t_max)
```

Clamping limits are set by `v52_cnc_max_focal_delta`, `v52_cnc_max_pp_offset_px`, `v52_cnc_max_rot_deg`, and `v52_cnc_max_t_delta`.

### 4. Optional temporal smoothing

When `v52_cnc_use_temporal_context = True`, a causal 1-D Conv1D is applied over `T` frames before the MLP, enforcing smooth camera drift without future leakage.

### 5. Auxiliary loss

A soft reprojection loss encourages geometric consistency while preventing over-correction:

```text
L_cnc = λ · Σ_v mask_v · ||π(K'_v, R'_v, t'_v, P_gt) - points_2d_v||_2
```

where `λ = v52_cnc_loss_weight` (default `0.0`).

---

## Inputs / Outputs

* **Inputs:** `points_2d (N,V,J,2)`, `confidences (N,V,J)`, `feat (N,V,J,d)`, `K (N,V,3,3)`, `R (N,V,3,3)`, `t (N,V,3)`, `view_mask (N,V)`.
* **Outputs:** `K_corrected (N,V,3,3)`, `R_corrected (N,V,3,3)`, `t_corrected (N,V,3)`, `cnc_loss` scalar.

---

## Integration into `OmniMultiViewFusionV5`

Instantiated in `__init__` when `use_v52_camera_noise_correction = True`. In `forward`, the correction is applied once after feature extraction and before triangulation:

```python
if self.use_v52_camera_noise_correction:
    K_corrected, R, t, cnc_loss = self.camera_noise_correction_v52(
        points_2d, confidences, feat, K_corrected, R, t, view_mask_flat
    )
```

Corrected cameras then flow through DLT, Gauss–Newton, v25/v45 geometry fusion, v46 sparse-view reliability, and physical-space alignment. No downstream module needs modification.

---

## Config flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v52_camera_noise_correction` | bool | `False` | Enable the module. |
| `v52_cnc_correct_intrinsics` | bool | `True` | Learn focal / principal point corrections. |
| `v52_cnc_correct_extrinsics` | bool | `True` | Learn rotation / translation corrections. |
| `v52_cnc_hidden` | int | `64` | MLP hidden dimension. |
| `v52_cnc_num_layers` | int | `2` | Number of MLP layers. |
| `v52_cnc_residual_gate_init` | float | `-6.0` | Initial residual gate. |
| `v52_cnc_max_focal_delta` | float | `0.05` | Max relative focal change. |
| `v52_cnc_max_pp_offset_px` | float | `5.0` | Max principal-point offset (px). |
| `v52_cnc_max_rot_deg` | float | `1.0` | Max rotation correction (deg). |
| `v52_cnc_max_t_delta` | float | `0.01` | Max translation correction (m). |
| `v52_cnc_loss_weight` | float | `0.0` | Auxiliary reprojection loss weight. |
| `v52_cnc_use_temporal_context` | bool | `True` | Use causal temporal smoothing. |

---

## Expected MPJPE impact

* **Clean WebBridge / H36M:** neutral to −1 mm.
* **Synthetic calibration noise (±2 px principal point, ±1% focal, ±0.5° rot, ±2 cm translation):** 3–6 mm improvement over v45/v51.
* **Sparse-view (v46/v51):** 1–3 mm gain on 2-view cases where camera errors are amplified.
* **Cross-domain (v48 / 3DPW):** modest gain from correcting systematic camera biases.

---

## 5-step implementation plan

1. **Module stub** (`motionflow_mv/fusion/camera_noise_correction_v52.py`): implement `CameraNoiseCorrectionV52` with confidence-weighted pooling, correction MLP, bounded updates, and optional causal temporal smoothing.
2. **Wiring in `OmniMultiViewFusionV5`**: add config flags, instantiate the module, and insert the correction call after feature extraction and before triangulation.
3. **Trainer hook**: expose `cnc_loss` and add it to the auxiliary loss dictionary with `v52_cnc_loss_weight`.
4. **Smoke test**: create `configs/benchmark_v52_camera_noise_correction_smoke.yaml` and run on RTX 4090; verify that with `v52_cnc_residual_gate_init = -6` the val_MPJPE matches the v45/v51 baseline.
5. **Full ablation**: add an A800 queue entry; evaluate with and without synthetic camera noise, comparing `MPJPE@k` and per-camera correction magnitudes.
