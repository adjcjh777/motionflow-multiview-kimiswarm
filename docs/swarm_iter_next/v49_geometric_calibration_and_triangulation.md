# v49: Geometric Calibration and Triangulation for MotionFlow-MultiView

**Status:** Design proposal / ready for swarm review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #166 (proposed)  
**Depends on:** v46-SVG (#160), v45-AGF  

---

## 1. Problem Statement

Current production runs (v25/v45–v48) treat camera calibration `(K, R, t)` as **given and fixed**, and rely on a single DLT solve followed by learned residuals. In practice this is fragile:

1. **Calibration drift / noise.** Real capture rigs have small intrinsics errors, lens distortion, or rig extrinsics drift that the fixed DLT cannot correct. The v21 neural bundle-adjustment block was disabled after regressing to 128 mm.
2. **Sparse/outlier 2D observations.** At 2–3 views, a single bad 2D joint can bias the whole triangulation; v46 sparse-view generalization improves robustness but still triangulates on fixed cameras.
3. **Static triangulation.** The DLT is a closed-form least-squares solution and does not learn from reprojection feedback, even when the model has strong evidence that a camera or view is unreliable.
4. **Self-evolution gap.** v37 self-critique and v46 reliability predict *view* trustworthiness, but the triangulation itself does not use those predictions to refine the underlying geometry or cameras.

**Goal for v49:** add a lightweight, identity-initialised geometric calibration & triangulation (GCT) block that jointly refines cameras and 3D pose from reprojection residuals, is robust to sparse/outlier views, and feeds a self-evolution feedback loop back into v46/v37 reliability.

---

## 2. Proposed Approach

v49 GCT sits **after** v25 multi-view geometry fusion and **before** v46 sparse-view reliability / v47 temporal aggregation:

```text
Input: 2D keypoints + cameras
        |
        v
[v25 MultiViewGeometryFusionV25]  -> initial triangulated pose P_init, cameras (K, R, t)
        |
        v
[v49 GeometricCalibrationTriangulationV49]
        |
        ├── Reprojection residual computation per view/joint
        ├── Lightweight camera-correction head (bounded update, zero-init)
        ├── Robust sparse-view triangulation refinement
        └── Returns refined pose P_ref and camera uncertainty U_cam
        |
        v
[v46 Sparse-View Generalization]  consumes refined pose + U_cam as extra reliability signal
[v47 Temporal Aggregation]        refines P_ref across time
```

### 2.1 Core ideas

1. **Bounded camera refinement.** Predict small corrections to `(K, R, t)` from reprojection residuals. Maximum updates are clamped to avoid the v21 divergence (e.g., `≤2°` rotation, `≤0.1 m` translation, `≤5%` focal scale). Final layer initialised to zero → identity at start.
2. **Residual-aware triangulation.** After optional camera refinement, re-triangulate each joint with per-view/joint weights derived from reprojection residuals and v46 reliability. Use a differentiable robust kernel (Geman-McClure/Huber) down-weighting outliers.
3. **Sparse-view guard.** Enforce `min_views ≥ 2`; if fewer views are active for a joint, keep the v25 estimate and set `U_cam` high.
4. **Camera uncertainty output.** Produce a per-view scalar `U_cam ∈ (0,1)` quantifying how much the camera was adjusted. This becomes an extra input to v46 reliability and the v37 self-critique gate.

### 2.2 Fit with v46–v48 and the overall pipeline

- **v46 Sparse-View Generalization:** v49 GCT provides a better initial triangulation and a camera-uncertainty signal that v46 can use to weight its reliability head. In sparse settings, the residual-aware re-triangulation directly improves `MPJPE@2`/`MPJPE@3`.
- **v47 Temporal Aggregation:** v47 operates on the refined pose `P_ref`; temporal smoothing is more stable because camera-induced drift has been partly removed before temporal fusion.
- **v48 Domain Generalization:** domain-specific calibration noise (studio rig vs. in-the-wild moving camera) can be handled by a domain-conditional camera-correction head, reusing the v48 `dataset_id` plumbing.
- **Overall multi-view pipeline:** the block is optional, zero-initialised, and only touches the triangulation/camera path; it does not change the ST transformer, physical loss, or domain losses.

---

## 3. Concrete Code-Level Changes

### 3.1 New module

**File:** `motionflow_mv/fusion/geometric_calibration_triangulation_v49.py`

```python
class GeometricCalibrationTriangulationV49(nn.Module):
    def __init__(
        self,
        n_joints: int = 17,
        camera_hidden: int = 64,
        max_rot_deg: float = 2.0,
        max_translation: float = 0.1,
        max_focal_scale: float = 0.05,
        max_principal_point_px: float = 10.0,
        robust_kernel: str = "geman_mcclure",
        robust_scale: float = 5.0,
        min_views: int = 2,
    ):
        ...

    def forward(
        self,
        points_2d: torch.Tensor,      # (B, T, V, J, 2)
        pred_3d_init: torch.Tensor,    # (B, T, J, 3)
        K: torch.Tensor,               # (B, T, V, 3, 3)
        R: torch.Tensor,               # (B, T, V, 3, 3)
        t: torch.Tensor,               # (B, T, V, 3)
        view_mask: torch.Tensor,       # (B, T, V)
        reliability: torch.Tensor | None = None,  # (B, T, V, J) from v46
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_3d_ref: (B, T, J, 3) refined 3D pose
            U_cam:       (B, T, V) camera uncertainty / correction magnitude
            aux_loss:    scalar robust reprojection loss
        """
```

Internal steps:
1. Compute per-view reprojection residuals from `pred_3d_init` and current cameras.
2. Predict bounded camera updates via a small MLP on residual statistics (mean/std over joints, weighted by reliability).
3. Re-triangulate using robust DLT with weights `w_vj ∝ reliability_vj / (1 + (residual_vj / robust_scale)^2)`.
4. Return refined pose, per-view camera uncertainty, and a small auxiliary Charbonnier reprojection loss.

### 3.2 Integration into `OmniMultiViewFusionV5`

**File:** `motionflow_mv/fusion/omniview_fusion_v5.py`

Constructor flags:

```python
use_geometric_calibration_triangulation_v49: bool = False,
v49_gct_camera_hidden: int = 64,
v49_gct_max_rot_deg: float = 2.0,
v49_gct_max_translation: float = 0.1,
v49_gct_max_focal_scale: float = 0.05,
v49_gct_robust_kernel: str = "geman_mcclure",
v49_gct_robust_scale: float = 5.0,
v49_gct_min_views: int = 2,
v49_gct_loss_weight: float = 0.05,
```

Forward hook (after v25 GN / before v46 reliability):

```python
if self.use_geometric_calibration_triangulation_v49:
    pred_3d_gn, U_cam, gct_loss = self.geometric_calibration_triangulation_v49(
        points_2d, pred_3d_gn, K_corrected, R, t,
        view_mask=view_mask, reliability=v46_reliability,
    )
    geom_loss_v25 = geom_loss_v25 + self.v49_gct_loss_weight * gct_loss
    # U_cam passed into v46/v37 for self-evolution feedback
```

### 3.3 Training-script flags

**File:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`

Add CLI arguments:

```python
parser.add_argument("--use_geometric_calibration_triangulation_v49", action="store_true")
parser.add_argument("--v49_gct_camera_hidden", type=int, default=64)
parser.add_argument("--v49_gct_max_rot_deg", type=float, default=2.0)
parser.add_argument("--v49_gct_max_translation", type=float, default=0.1)
parser.add_argument("--v49_gct_max_focal_scale", type=float, default=0.05)
parser.add_argument("--v49_gct_robust_kernel", type=str, default="geman_mcclure")
parser.add_argument("--v49_gct_robust_scale", type=float, default=5.0)
parser.add_argument("--v49_gct_min_views", type=int, default=2)
parser.add_argument("--v49_gct_loss_weight", type=float, default=0.05)
```

### 3.4 New assets

- `configs/benchmark_v49_gct_smoke.yaml` — smoke config.
- `scripts/run_v49_gct_smoke_local_4090.sh` — smoke script.
- `tests/test_geometric_calibration_triangulation_v49.py` — unit tests for identity at init, view masking, and bounded camera updates.

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|------|------------|
| **Camera-correction head diverges** like v21 neural BA. | Zero-init final layer, clamp updates, detach the camera update decision from gradients, and gate on reprojection improvement only. |
| **Sparse views (<2) cause ill-posed triangulation.** | Hard `min_views=2` guard; fall back to v25 estimate and high `U_cam`. |
| **Robust kernel down-weights good views.** | Initialise `robust_scale` from median reprojection residual on the first batch; monitor per-view weight histograms. |
| **Gradient instability through DLT + camera correction.** | Stop-grad through the camera-correction MLP's input residuals for the first epoch; use `torch.linalg.lstsq` with ridge regularisation. |
| **Conflicts with v25 geometry bundle adjustment** (placeholder). | Disable `v25_use_geometry_bundle_adjustment` when v49 is active; they target the same camera-refinement role. |
| **Self-evolution loop amplifies wrong reliability.** | Detach `U_cam` from the pose loss so it only guides v46/v37, not the other way around, until smoke validates stability. |

---

## 5. Success Metrics and Recommended Experiments

### 5.1 Primary metrics

- **val_MPJPE** on mixed H36M + MPI-INF-3DHP val.
- **Reprojection error** on val (should be non-increasing after v49 block).
- **Sparse-view robustness:** `MPJPE@2`, `MPJPE@3`, `MPJPE@4` via `experiments/eval_variable_views.py`.
- **Camera-perturbation robustness:** apply `cam_aug_schedule extended_curriculum` and measure gap vs. clean MPJPE.

### 5.2 Smoke experiment

| Stage | Hardware | Config | Expected outcome |
|-------|----------|--------|------------------|
| Smoke | RTX 4090 | `configs/benchmark_v49_gct_smoke.yaml` | val_MPJPE < 80 mm, no NaN/OOM, `U_cam` histogram within `[0, 0.1]` for >80% of views (camera head starts near identity). |

Example smoke command:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --config configs/benchmark_v49_gct_smoke.yaml \
  --use_geometric_calibration_triangulation_v49 \
  --v49_gct_loss_weight 0.05
```

### 5.3 Full experiment

| Stage | Hardware | Config | Expected outcome |
|-------|----------|--------|------------------|
| Full | A800-D | v46/v48 base + v49 GCT | ≥1 mm improvement over v46 baseline on full-view val_MPJPE; ≥5% relative improvement on `MPJPE@2/3`; no reprojection regression. |

### 5.4 Ablations

1. `v49_gct_no_camera_correction` — robust triangulation only, no camera MLP.
2. `v49_gct_no_robust_kernel` — plain confidence-weighted DLT re-triangulation.
3. `v49_gct_high_clamp` — double the camera update bounds to test sensitivity.
4. `v49_gct_with_v46` vs. `v49_gct_without_v46` — measure the value of feeding `U_cam` into v46 reliability.

---

## 6. Self-Evolution Feedback Loop

v49 GCT closes a geometry-level self-evolution loop:

1. **Forward pass:** v25 produces an initial pose and fixed cameras. v49 computes reprojection residuals and refines both.
2. **Camera uncertainty signal:** `U_cam` encodes how much each camera had to be corrected. This is passed to v46 sparse-view reliability and the v37 self-critique gate as an extra input.
3. **Reliability re-weighting:** in subsequent iterations, v46 reliability can down-weight views with high `U_cam` before the geometry head even sees them.
4. **Supervision:** the robust reprojection loss from v49 supervises the camera-correction head directly, so the model learns to diagnose and fix its own calibration errors from data.

This is a narrower, more geometric analogue of the v37/v39 self-evolution loop: instead of predicting abstract view reliability, v49 corrects the actual camera geometry and feeds the magnitude of that correction forward.

---

## 7. Next Steps

1. Wait for v46-SVG smoke results (#160) to land a stable sparse-view baseline.
2. Implement `GeometricCalibrationTriangulationV49` and unit tests.
3. Wire flags into `OmniMultiViewFusionV5` and the trainer.
4. Smoke on RTX 4090 and verify bounded camera updates + no reprojection regression.
5. Queue full A800 run on top of the best v46/v48 checkpoint.
