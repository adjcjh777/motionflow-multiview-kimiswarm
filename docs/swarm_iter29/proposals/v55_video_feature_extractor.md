# v55 Video Feature Extractor (VFE)

## 1. Module name and one-line purpose

**Module:** `VideoFeatureExtractorV55` → `motionflow_mv/fusion/video_feature_extractor_v55.py`

**One-liner:** A lightweight, identity-at-init per-view spatiotemporal feature refiner that improves the 2D keypoint + confidence representation before triangulation, making downstream geometry fusion (v45/v25) and uncertainty-weighted triangulation (v52) more robust, especially when only a few views are available.

## 2. Placement in the OmniMultiViewFusionV5 forward pass

The module sits **after the raw 2D keypoint/confidence inputs and before the v45/v25 geometry-fusion block**. It refines the per-view, per-joint tokens that are fed into the rest of the pipeline; all later blocks (v52 UWT, v53 PSC, v54 PSC-v2, v47/v49 temporal, v50 SEFH) operate on the same interfaces as before.

```text
points_2d, confidences, K, R, t
    ↓
v55 VideoFeatureExtractorV55 → refined_2d, refined_conf, vfe_loss
    ↓
v45 / v25 geometry fusion → pred_3d_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt
    
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc
    
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

## 3. Inputs, outputs, and shapes

**Inputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `points_2d` | `(B, T, V, J, 2)` | Raw 2D keypoint coordinates (pixels or normalized). |
| `confidences` | `(B, T, V, J)` | Raw per-keypoint confidence scores. |
| `K` | `(B, V, 3, 3)` | Intrinsic matrices. |
| `R` | `(B, V, 3, 3)` | Camera rotations matrices. |
| `t` | `(B, V, 3)` | Camera translation vectors. |
| `view_mask` | `(B, T, V)` | Binary mask for visible/valid views (optional). |

**Outputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `refined_2d` | `(B, T, V, J, 2)` | Gated residual refinement of `points_2d`. |
| `refined_conf` | `(B, T, V, J)` | Gated residual refinement of `confidences`. |
| `vfe_loss` | `()` | Auxiliary reprojection-consistency loss (scalar). |

At initialization, `refined_2d == points_2d` and `refined_conf == confidences` up to numerical error because the residual output projection is zero-initialized and the gate is closed (`σ(-6.0) ≈ 0.0025`).

## 4. Architecture

### 4.1 Core blocks

1. **Input token projection.** Concatenate `points_2d` and `confidences` and project to `v55_vfe_hidden` dimensions (`Linear(3 → D)`).
2. **Camera embedding.** Flatten `K`, `R`, `t` and project to `D`; broadcast across `(T, J)` and add to the per-view tokens. Controlled by `v55_vfe_use_camera_embed`.
3. **Per-view temporal convolution.** Two causal .k-3 1-D convolutions over `T` (depth = `v55_vfe_n_layers`, hidden `D`) with GroupNorm + GELU + dropout.
4. **Cross-joint attention.** A single multi-head self-attention block (`v55_vfe_num_heads` heads, hidden `D`) over the `J` dimension per `(B, T, V)` token, with a residual connection and layer-norm. This lets nearby joints borrow information without a full skeleton graph.
5. **Output projection.** Two separate zero-initialized linear heads:
   - `Δpoints_2d`: `(D → 2)`
   - `Δconf`: `(D → 1)`
6. **Residual gate.** A learnable scalar gate initialized to `v55_vfe_residual_gate_init = -6.0`.

```
refined_2d  = points_2d  + σ(gate) · Δpoints_2d
refined_conf = sigmoid(confidences + σ(gate) · Δconf)   # if raw conf is logit, else add then clip
```

### 4.2 Auxiliary loss

`vfe_loss` is a **geometry-aware reprojection consistency** term: triangulate a reference 3D pose from `refined_2d` using the existing v45/v25 triangulation block, reproject it to each view, and measure the weighted 2D error. At identity initialization, `refined_2d == points_2d`, so the loss is zero. The total loss is:

```
L_total = L_base + v55_vfe_loss_weight * vfe_loss
```

with `v55_vfe_warmup_epochs` epochs of zero weight before the auxiliary loss contributes.

### 4.3 Identity-at-init mechanism

- Output layers `Δpoints_2d` and `Δconf` are zero-initialized.
- Gate logit initialized to `-6.0`, so the effective blend weight is `σ(-6.0) ≈ 0.0025`.
- `v55_vfe_loss_weight` defaults to `0.01` and is ramped in after `v55_vfe_warmup_epochs = 0` epochs.
- Loading a v54 checkpoint with v55 enabled therefore leaves the baseline MPJPE unchanged (`< 0.1 mm`).

## 5. Expected MPJPE impact and main risks

**Expected impact**

- **Full views:** `−0.8 mm` to `−2.0 mm` MPJPE.
- **Sparse views (`@2` / `@3`):** `−2.0 mm` to `−4.0 mm`, where better per-joint 2D features compensate for missing views.
- The largest gains are expected on H36M and WebBridge, where the raw 2D detector output is noisy and temporally inconsistent.

**Main risks**

| Risk | Symptom | Mitigation |
|------|-----------|------------|
| **Over-smoothing fast motion** | Wrist/ankle jitter reduction turns into motion blur; MPJPE rises on fast sequences. | Keep temporal kernel small (3), use causal padding, and make temporal refinement optional via `v55_vfe_use_temporal`. |
| **Camera embedding overfits to calibration errors** | Refined 2D absorbs wrong camera parameters and biases triangulation. | Gate init `−6.0`; camera embedding can be disabled with `v55_vfe_use_camera_embed=False`; zero-init output layers. |
| **Downstream triangulation becomes unstable** | v52 UWT weights diverge because refined 2D changes the geometry. | Bound the 2D residual magnitude in pixel space; use the residual gate as a conservative throttle. |
| **Identity-at-init regression** | Loading v54 with v55 enabled changes baseline MPJPE by `>0.1 mm`. | Unit-test `||refined_2d − points_2d||_∞ < 1e-4`; assert gate init `−6.0`; assert `vfe_loss == 0` at init. |

## 6. Smoke acceptance criteria

Run `bash scripts/run_v55_video_feature_extractor_smoke_local_4090.sh` on the local RTX 4090 and verify:

1. **No regression:** `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke seed and manifest.
2. **Identity-at-init:** loading the best available v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
3. **Stability:** no NaN, Inf, or OOM through at least one full epoch.
4. **Sparse-view sanity:** `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.
5. **Finite residuals:** max absolute `Δpoints_2d` and `Δconf` are finite and bounded by the configured pixel/logit clip.
6. **Loss sanity:** `vfe_loss` is zero (within `1e-5`) at initialization and remains finite after the first training step.

## 7. Required new files and files to modify

**New files**

- `motionflow_mv/fusion/video_feature_extractor_v55.py` — `VideoFeatureExtractorV55` module.
- `configs/benchmark_v55_video_feature_extractor_smoke.yaml` — smoke config, copied from `benchmark_v54_psc_v2_smoke.yaml` with v55 flags enabled.
- `scripts/run_v55_video_feature_extractor_smoke_local_4090.sh` — smoke launch script that warm-starts from the best v54 checkpoint.
- `tests/test_video_feature_extractor_v55.py` — unit tests for identity-at-init, residual bounds, camera-embedding optional path, and gradient flow.

**Files to modify**

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag `use_v55_video_feature_extractor`, instantiate `VideoFeatureExtractorV55` when enabled, insert the call after the raw 2D/confidence inputs and before v45/v25 geometry fusion, and add `vfe_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — aggregate `loss_dict["v55_vfe"]` with weight `v55_vfe_loss_weight` and warmup guard.
- `scripts/launch_v33_a800_queue.py` — add A800 full-run entry `v55_video_feature_extractor_on_v54`.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_video_feature_extractor` | bool | `False` | Master toggle |
| `v55_vfe_hidden` | int | `64` | Hidden dimension for temporal/attention blocks |
| `v55_vfe_n_layers` | int | `2` | Number of temporal conv layers |
| `v55_vfe_num_heads` | int | `4` | Attention heads in cross-joint block |
| `v55_vfe_temporal_window` | int | `None` | Optional fixed temporal window; `None` uses full clip |
| `v55_vfe_identity_init` | bool | `True` | Zero-initialize output projection layers |
| `v55_vfe_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_vfe_use_temporal` | bool | `True` | Enable temporal conv path |
| `v55_vfe_use_camera_embed` | bool | `True` | Inject camera parameters into tokens |
| `v55_vfe_loss_weight` | float | `0.01` | Multiplier on `vfe_loss` |
| `v55_vfe_warmup_epochs` | int | `0` | Epochs before `vfe_loss` contributes to total loss |
| `v55_vfe_2d_residual_clip` | float | `10.0` | Max pixel refinement per joint |
| `v55_vfe_conf_residual_clip` | float | `1.0` | Max confidence logit refinement |

---

**Paper alignment:** The module directly strengthens the *multi-view fusion* narrative by improving the per-view 2D observations that feed triangulation. It is a low-risk, residual, identity-at-init block that preserves all existing v52–v54 calibration machinery while giving the largest sparse-view gains of the v55 candidate set.
