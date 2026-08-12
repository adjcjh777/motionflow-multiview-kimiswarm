# v33: Geometry-Aware Neural Bundle Adjustment / Geometry-Aware Refinement

**Slug:** `neural_bundle_adjustment`  
**Status:** Design proposal — no source files modified.  
**Related work:** v21 `NeuralBundleAdjustment` (`motionflow_mv/fusion/neural_bundle_adjustment_v21.py`), v25 `MultiViewGeometryFusionV25` (`motionflow_mv/fusion/multiview_geometry_fusion_v25.py`), `DifferentiableBundleAdjustment` (`motionflow_mv/fusion/differentiable_bundle_adjustment.py`).

---

## 1. Problem statement and motivation

The v31/v32 stack already combines deformable cross-view attention, multi-view geometry fusion (v25), hierarchical encoders (v30/v31), and physical-space losses. However, two geometry-oriented blocks are still under-utilised or unsafe:

* **v21 neural bundle adjustment regressed to 128.27 mm** on WebBridge and is currently disabled in production runs. The neural camera-correction head saw noisy initial points, could move good cameras to explain bad points, and had no geometric guard against increasing reprojection error.
* **v25 geometry bundle adjustment is a placeholder** — `MultiViewGeometryFusionV25` stores `use_geometry_bundle_adjustment=True` but does not refine cameras; only ray tokens, geometry attention, and the learned depth-proposal head are active.

**v33** therefore proposes a *geometry-aware neural bundle-adjustment* block that fuses the analytic structure update from v21/differentiable-BA with the ray-geometry machinery from v25, and wraps it with a residual-improvement gate so it cannot destabilise training.

---

## 2. Proposed architecture changes

### 2.1 New module

**File:** `motionflow_mv/fusion/geometry_aware_neural_bundle_adjustment_v33.py`

**Class:** `GeometryAwareNeuralBundleAdjustmentV33`

Forward signature:

```python
pred_3d_ref, K_ref, R_ref, t_ref, geom_loss = module(
    pred_3d_init,   # (B, T, J, 3)
    points_2d,      # (B, T, V, J, 2)
    K,              # (B, T, V, 3, 3)
    R,              # (B, T, V, 3, 3)
    t,              # (B, T, V, 3)
    confidence,     # (B, T, V, J)
    view_mask=None, # (B, T, V)
)
```

Core loop (identity-initialised, bounded updates):

1. **Structure-only Gauss-Newton step** — reuse `_project_and_jacobian` from `differentiable_bundle_adjustment.py`, solve `(J^T W J + λI) ΔX = J^T W r`, clamp `ΔX`, and **detach** the output before the camera head sees it.
2. **Geometry-aware camera descriptor** — for each view concatenate:
   * mean/std of reprojection residual,
   * intrinsics (`fx, fy, cx, cy, skew`),
   * axis-angle rotation (3-DOF),
   * translation,
   * total per-view weight,
   * ray-intersection quality with every other view (mean/std of the shortest 3D ray distance and baseline-angle cosine, computed via the same helpers used by v25).
3. **Neural camera-correction head** — small MLP (or 1-layer transformer over joints) predicting bounded updates to focal length, principal point, rotation (axis-angle), and translation. Final layer initialised to zero.
4. **Residual-improvement gate** — accept a camera update only if it does not increase the per-view mean reprojection error (with a small tolerance). The gate decision is detached so the MLP still receives gradients from accepted updates.
5. **Optional auxiliary geometry loss** — Charbonnier reprojection loss on the refined output, weighted by `v33_geom_loss_weight`.

### 2.2 Integration into `OmniMultiViewFusionV5`

**File:** `motionflow_mv/fusion/omniview_fusion_v5.py`

Add constructor flags:

```python
use_geometry_aware_bundle_adjustment_v33: bool = False,
v33_n_iters: int = 2,
v33_camera_hidden: int = 128,
v33_max_point_update: float = 0.05,
v33_max_rotation_deg: float = 2.0,
v33_max_translation: float = 0.1,
v33_max_focal_scale: float = 0.05,
v33_max_principal_point_px: float = 10.0,
v33_geom_loss_weight: float = 0.05,
```

Instantiation (mirrors v21):

```python
self.use_geometry_aware_bundle_adjustment_v33 = use_geometry_aware_bundle_adjustment_v33
if self.use_geometry_aware_bundle_adjustment_v33:
    self.geometry_aware_neural_bundle_adjustment_v33 = GeometryAwareNeuralBundleAdjustmentV33(
        n_iters=v33_n_iters,
        camera_hidden=v33_camera_hidden,
        max_point_update=v33_max_point_update,
        max_rotation_deg=v33_max_rotation_deg,
        max_translation=v33_max_translation,
        max_focal_scale=v33_max_focal_scale,
        max_principal_point_px=v33_max_principal_point_px,
        geom_loss_weight=v33_geom_loss_weight,
    )
```

Forward hook — insert **after** the adaptive Gauss-Newton refinement (`pred_3d_gn`) and **before** v25 geometry fusion / residual refinement:

```python
if self.use_geometry_aware_bundle_adjustment_v33 and ...:
    pred_3d_gn, K_corrected, R, t, v33_geom_loss = \
        self.geometry_aware_neural_bundle_adjustment_v33(
            pred_3d_gn,
            points_2d.view(B, T, V, J, 2),
            K_corrected.view(B, T, V, 3, 3),
            R.view(B, T, V, 3, 3),
            t.view(B, T, V, 3),
            confidence=confidences.view(B, T, V, J),
            view_mask=view_mask_flat.view(B, T, V),
        )
    pred_3d_gn = pred_3d_gn.view(B * T, J, 3)
    geom_loss_v25 = geom_loss_v25 + v33_geom_loss
```

### 2.3 Training-script flags

**File:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`

Add CLI arguments and pass them through `build_model_from_args` exactly like v21/v25 toggles:

```python
parser.add_argument("--use_geometry_aware_bundle_adjustment_v33", action="store_true",
                    help="Use v33 geometry-aware neural bundle-adjustment refiner")
parser.add_argument("--v33_n_iters", type=int, default=2)
parser.add_argument("--v33_camera_hidden", type=int, default=128)
parser.add_argument("--v33_geom_loss_weight", type=float, default=0.05)
```

---

## 3. Training command / ablation flags

### Smoke test (local RTX 4090, ~5 min)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 \
  --use_geometry_aware_bundle_adjustment_v33 \
  --v33_n_iters 2 --v33_geom_loss_weight 0.05
```

### Full run (A800-D / local queue)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
  --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
  --use_hierarchical_multiview_v30 --v30_n_part_layers 2 \
  --use_geometry_aware_bundle_adjustment_v33 \
  --v33_n_iters 2 --v33_camera_hidden 128 --v33_geom_loss_weight 0.05 \
  --d 64 --residual_hidden 128 --n_st_layers 2 --n_joint_layers 1 --n_heads 4 \
  --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 \
  --output outputs/omniview_fusion_v33_geometry_aware_nba.pth
```

### Ablations

| Flag | Purpose |
|------|---------|
| `--use_geometry_aware_bundle_adjustment_v33` | Master toggle for the new block. |
| `--v33_n_iters 0` | Disable iterative refinement (structure-only/no-op). |
| `--v33_geom_loss_weight 0.0` | Disable the auxiliary reprojection loss. |
| `--use_neural_bundle_adjustment_v21` (instead) | Compare directly against the original v21 block. |
| `--no_v25_use_geometry_bundle_adjustment` / omit v25 | Test v33 without v25 ray-geometry context. |
| `--cam_aug_schedule extended_curriculum` | Stress-test camera-correction head. |

---

## 4. Expected metrics and baseline to beat

Baseline is the current v31/v32 production stack (v25 + v30 hierarchical + physical loss), which reports mixed WebBridge/H36M/MPI val_MPJPE in the **21–28 mm** range on A800-D (e.g. v29o 21.54 mm, v29u 27.58 mm).

**Success criteria for v33:**

| Metric | Target |
|--------|--------|
| **Mixed val_MPJPE** | ≤ baseline; a 1–2 mm improvement would validate the camera-refinement head. |
| **PA-MPJPE** | Match or beat the baseline (currently ~5–6 mm on MPI-INF-3DHP clean). |
| **Camera-perturbation robustness** | With `--cam_aug_schedule extended_curriculum`, gap vs. clean MPJPE reduced by ≥10% relative to the v31/v32 run. |
| **Variable-view inference** | 2-view subset MPJPE within 15% of the 4-view result (leverages existing variable-view training). |
| **Reprojection stability** | Mean val reprojection error non-increasing after the v33 block; no NaN/Inf or >5 mm regression. |
| **GPU memory/time** | <5% throughput drop vs. the same run without v33. |

---

## 5. Risks / unknowns

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| **v21-style regression** | The camera head can still worsen cameras if the residual gate or detach is wrong. | Keep the structure update detached from the camera head; hard/soft gate on reprojection improvement; identity zero-initialisation. |
| **Gradient instability through Gauss-Newton** | `torch.linalg.solve` inside the structure step can backpropagate large gradients. | Clamp updates, add LM damping, and optionally stop-grad into the GN solver for the first epochs. |
| **Variable-view masking edge cases** | Ray-intersection quality is pairwise; dropping views changes pairwise statistics. | Compute descriptors only over active views using `view_mask`; test with `variable_view_min_views=2`. |
| **Extra compute/memory** | Ray geometry + camera head adds cost on top of v25/v30. | Keep the MLP small (`camera_hidden=128`); limit `v33_n_iters=2`; profile throughput on the smoke run. |
| **Geometry descriptor not helpful** | The v25 ray quality features may not add signal beyond reprojection residuals. | Ablation flag `--v33_no_geometry_descriptor` to measure contribution. |
| **Interaction with v29/v30 hierarchical encoders** | The block sits after GN but before v25; hierarchical features may already encode similar geometry. | Compare against a run that has `--use_hierarchical_multiview_v30 --use_geometry_aware_bundle_adjustment_v33` and one without hierarchy. |

---

## 6. Files that would be touched (implementation note)

* `motionflow_mv/fusion/geometry_aware_neural_bundle_adjustment_v33.py` — new module.
* `motionflow_mv/fusion/omniview_fusion_v5.py` — add v33 flags and forward hook.
* `experiments/train_omniview_fusion_v5_webbridge_multi.py` — add CLI flags and pass them into `build_model_from_args`.
* `tests/test_geometry_aware_neural_bundle_adjustment_v33.py` — new smoke test for shape, orthogonality, backward pass, and residual gate.
