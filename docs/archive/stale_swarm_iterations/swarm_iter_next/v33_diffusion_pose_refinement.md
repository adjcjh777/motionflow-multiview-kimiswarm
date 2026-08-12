# v33: Diffusion-based 3D pose denoising/refinement

**Slug:** `diffusion_pose_refinement`  
**Author:** v33 diffusion swarm agent  
**Status:** Design proposal  
**Depends on:** `v5`/`v25`/`v30`/`v31`/`v32` stack, existing `v20` diffusion refiner prototype (`motionflow_mv/fusion/diffusion_pose_refiner_v20.py`)

---

## 1. Problem statement and motivation

`OmniMultiViewFusionV5` currently ends with a deterministic residual MLP:

```python
# motionflow_mv/fusion/omniview_fusion_v5.py ~1100-1103
residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)
delta = self.residual_mlp(residual_input)
pred_3d = pred_3d_gn + delta
```

This works for small, structured corrections, but it is a *point estimate* and struggles with the multi-modal residual errors produced by:

* **Few/occluded views:** variable-view training drops to 2–4 views; the optimal correction can be multi-modal.
* **Outlier views / joint occlusion:** the robust DLT path down-weights bad views, but the final residual still contains structured uncertainty.
* **Geometry-fusion residual:** after `v25`/`v30` geometry fusion the remaining error is small but correlated across joints and time.

A lightweight diffusion-based refiner can model this residual distribution, and existing `v20` code shows the project already has a working prototype. However, `v20` is not wired into the current `v32` queue and lacks geometry/visibility conditioning. **v33 proposes a geometry- and visibility-aware temporal diffusion refiner that replaces the deterministic MLP as a drop-in, gated head.**

---

## 2. Proposed architecture changes

### 2.1 New module

**File:** `motionflow_mv/fusion/diffusion_pose_refiner_v33.py`

```text
DiffusionPoseRefinerV33(
    j: int = 17,
    in_dim: int = 64,                # dimension of feat_pooled per joint
    num_diffusion_steps: int = 50,   # training steps T
    num_inference_steps: int = 3,    # fast DDIM/DPMSolver steps at eval
    residual_hidden: int = 128,
    n_heads: int = 4,
    temporal_window: int = 3,        # must be odd; 1 => per-frame
    use_geometry_conditioning: bool = True,
    use_visibility_conditioning: bool = True,
    use_temporal_conditioning: bool = True,
    beta_schedule: str = "cosine",
)
```

Key improvements over `v20`:

| Aspect | v20 | v33 |
|--------|-----|-----|
| Condition signal | `feat_pooled` only | `feat_pooled` + per-joint visibility + geometry-aware ray/depth scores from `v25` |
| Temporal reasoning | None | Optional `temporal_window` self-attention over residuals |
| Fast inference | 5-step DDPM | 3-step DDIM / DPMSolver single-shot |
| View robustness | Implicit | Explicit visibility gating: per-joint visible-view fraction feeds the denoiser |
| Warm start | Final layer zeroed | Same; identity at init so baseline is preserved |

### 2.2 What the denoiser predicts

At the hook after Gauss-Newton / `v25`/`v30` geometry fusion:

```python
# Current residual MLP (to be replaced when flag is active)
residual_target = y - pred_3d_gn          # residual to denoise
pred_3d, diff_loss = self.diffusion_refiner_v33(
    pose_init=pred_3d_gn,                 # (B, T, J, 3)
    feat=feat_pooled,                     # (B*T, J, d)
    visibility=visibility,                # (B, T, V, J)
    view_mask=view_mask,                  # (B, T, V)
    train_targets=y if self.training else None,
)
```

The diffusion process runs on the residual `δ = y - pred_3d_gn`. During training:

1. Sample `t ~ Uniform(0, T-1)`.
2. Add noise: `δ_t = sqrt(α_t) * δ + sqrt(1 - α_t) * ε`.
3. The denoiser predicts `ε` conditioned on `δ_t`, `pred_3d_gn`, `feat_pooled`, per-joint visibility, and optional temporal neighbours.
4. Loss: `diff_loss = MSE(predicted_ε, ε)`.

At inference, run fast deterministic sampling for `num_inference_steps` steps and add the final residual to `pred_3d_gn`.

### 2.3 Geometry / visibility feature source

`v33` re-uses tensors already computed in `OmniMultiViewFusionV5.forward`:

* `feat_pooled`: mean-pooled per-view ST features, shape `(B*T, J, d)`.
* `visibility`: already returned by the model, shape `(B, T, V, J)`.
* `weights`: per-view triangulation weights, shape `(B, T, V, J)`.

A tiny on-module `GeometryConditioningV33` projects the following per-joint scalar fields to `geom_dim`:

* `visible_view_fraction` (from `visibility` and `view_mask`)
* mean and entropy of triangulation `weights` across views
* ray-intersection quality score (recomputed cheaply from `points_2d`, `K`, `R`, `t`)

No change to `v25`/`v30` internals is required.

### 2.4 Integration point

Replace the deterministic residual MLP block in `motionflow_mv/fusion/omniview_fusion_v5.py` (lines ~1086–1103) with a new gated branch:

```python
if self.use_diffusion_refiner_v33 and self.diffusion_refiner_v33 is not None:
    pred_3d, diff_loss = self.diffusion_refiner_v33(
        pose_init=pred_3d_gn.view(B, T, J, 3),
        feat=feat_pooled,                       # (B*T, J, d)
        visibility=visibility.view(B, T, V, J),
        view_mask=view_mask_flat.view(B, T, V),
        train_targets=y if self.training else None,
    )
    pred_3d = pred_3d.view(B * T, J, 3)
    epi_loss = epi_loss + self.v33_diffusion_loss_weight * diff_loss
else:
    # existing deterministic MLP (default) or v20 path
    residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)
    delta = self.residual_mlp(residual_input)
    pred_3d = pred_3d_gn + delta
```

### 2.5 New constructor flags in `OmniMultiViewFusionV5`

Add next to the existing `use_diffusion_refiner_v20` block:

```python
use_diffusion_refiner_v33: bool = False,
v33_diffusion_loss_weight: float = 0.1,
v33_diffusion_num_steps: int = 50,
v33_diffusion_num_inference_steps: int = 3,
v33_diffusion_residual_hidden: int = 128,
v33_diffusion_temporal_window: int = 3,
v33_diffusion_use_geometry_conditioning: bool = True,
v33_diffusion_use_visibility_conditioning: bool = True,
v33_diffusion_use_temporal_conditioning: bool = True,
```

---

## 3. Training command / ablation flags

Smoke test (RTX 4090 / CPU):

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_diffusion_refiner_v33 \
  --v33_diffusion_loss_weight 0.1 \
  --v33_diffusion_num_inference_steps 3
```

Full mixed-dataset run on the current `v32` baseline:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
  --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 \
  --v25_use_geometry_attention --v25_use_learned_depth_triangulation \
  --v25_use_geometry_bundle_adjustment \
  --use_hierarchical_multiview_v30 --v30_n_part_layers 2 \
  --use_camera_view_embedding --use_set_view_aggregator \
  --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
  --outlier_view_prob 0.3 \
  --use_diffusion_refiner_v33 \
  --v33_diffusion_loss_weight 0.1 \
  --v33_diffusion_num_steps 50 \
  --v33_diffusion_num_inference_steps 3 \
  --v33_diffusion_residual_hidden 128 \
  --v33_diffusion_temporal_window 3 \
  --v33_diffusion_use_geometry_conditioning \
  --v33_diffusion_use_visibility_conditioning \
  --num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --clip_len 9 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --epochs 20 --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 \
  --output outputs/omniview_fusion_v33_diffusion_refiner.pth
```

Ablation flags to isolate the contribution of each component:

```bash
# Baseline: deterministic residual MLP (no diffusion)
--use_diffusion_refiner_v33 false

# Per-frame diffusion only (no temporal)
--use_diffusion_refiner_v33 --v33_diffusion_temporal_window 1

# No geometry/visibility conditioning
--use_diffusion_refiner_v33 \
  --no_v33_diffusion_use_geometry_conditioning \
  --no_v33_diffusion_use_visibility_conditioning

# Fast single-step inference (for latency benchmark)
--use_diffusion_refiner_v33 --v33_diffusion_num_inference_steps 1
```

---

## 4. Expected metrics and baseline to beat

Use the current `v32` deterministic `OmniMultiViewFusionV5` run with `--use_hierarchical_multiview_v30` as the baseline.

| Metric | Baseline (v32) | v33 target | How to measure |
|--------|----------------|------------|----------------|
| `val_MPJPE` | ~20–22 mm on mixed H36M/MPI | **-5 % to -10 %** (~18–20 mm) | `experiments/train_omniview_fusion_v5_webbridge_multi.py` validation log |
| `val_MPJPE@2views` | highest error | **-8 % to -15 %** | variable-view eval with `variable_view_max_views=2` |
| `val_MPJPE@14views` | near saturated | **-2 % to -5 %** | variable-view eval with full 14 views |
| PA-MPJPE | track with baseline | maintain or improve | Procrustes-aligned MPJPE in eval |
| Inference latency | MLP forward | < 1.5× baseline | `num_inference_steps=3` |

**Baseline to beat:** the deterministic residual MLP branch in the current `v32` queue. A successful v33 experiment should lower the first-epoch `val_MPJPE` below the deterministic baseline and not regress full-run PA-MPJPE.

---

## 5. Risks / unknowns

| Risk | How to detect | Mitigation |
|------|---------------|------------|
| Diffusion loss destabilises training | `diff_loss` >> `mpjpe_loss` in first 500 steps | Start with `v33_diffusion_loss_weight=0.1`; clamp residual to `[-0.2, 0.2]` m; zero-initialise output projection |
| Inference too slow for 20-epoch full runs | Latency >2× baseline | Default to `num_inference_steps=1` (single deterministic step) or use DDIM/DPMSolver-3 |
| Overfits to MPI 14-view layout | H36M val error rises while MPI improves | Add `--domain_aware_view_curriculum`; freeze refiner for first epoch |
| Geometry feature extractor is noisy | Ablating conditioning shows no benefit | Make extractor tiny (<0.1 M params); fall back to `feat_pooled` + visibility only |
| v20 code path collision | Two diffusion flags coexist | v33 supersedes v20; assert `not (use_diffusion_refiner_v20 and use_diffusion_refiner_v33)` |
| Diffusion head has large memory cost for T>13 | OOM on `clip_len=21` | Use `temporal_window=1` for long clips; attention over `J` only keeps memory low |

---

## 6. Minimal validation plan

1. **Module smoke:** `python motionflow_mv/fusion/diffusion_pose_refiner_v33.py` (add `if __name__ == "__main__":` block).
2. **Unit test:** `tests/test_diffusion_pose_refiner_v33.py` covering:
   * Forward shape `(B, T, J, 3)` in inference mode.
   * Training returns `(refined, loss)` and loss backprops.
   * Identity at init when output projection is zeroed.
   * Variable view mask handling (`view_mask` with missing views).
3. **Pipeline smoke:** run `--smoke` with `--use_diffusion_refiner_v33`.
4. **Ablated full run:** run v33 vs. deterministic MLP baseline on the `v32` flag set with at least one full A800 GPU cycle.

---

## 7. References

* `motionflow_mv/fusion/omniview_fusion_v5.py` — current residual MLP hook (`pred_3d_gn` + `feat_pooled`).
* `motionflow_mv/fusion/diffusion_pose_refiner_v20.py` — prior diffusion prototype.
* `docs/proposals/v27_diffusion_refinerv2.md` — earlier geometry-aware temporal diffusion design.
* `docs/proposals/v20_diffusion_refinement.md` — original v20 design doc.
* `scripts/launch_v32_a800_queue.py` — current flag set and baseline queue.
