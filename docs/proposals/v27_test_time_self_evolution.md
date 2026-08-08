# v27: Test-Time Self-Evolution via Geometric Self-Consistency

**Task identifier:** `design_v27_test_time_self_evolution`  
**Depends on:** v25 (`docs/proposals/v25_multiview_geometry_fusion.md`), v26 (`docs/proposals/v26_temporal_geometry_fusion.md`)  
**Status:** Design / Candidate direction

> **Scope note:** The label "test-time self-evolution" is intentionally open. This document proposes the simplest concrete interpretation: a *test-time iterative self-consistency loop* that reuses the learned v25/v26 triangulation head and v22 bone-length prior. If the broader interpretation (online model-update at inference) is intended, it should be deferred to a later iteration because it requires checkpoint mutability and careful validation plumbing that the current pipeline does not have.

---

## 1. Problem

v25 and v26 are **single-shot feed-forward blocks**: the model produces one refined 3D pose and stops. The current `MultiViewGeometryFusionV25` / `TemporalGeometryFusionV26` blocks already reason with rays, epipolar constraints, and depth proposals, but they do not verify whether their own output is geometrically self-consistent. Concretely:

* **Outlier views dominate few-view settings.** At 2–4 views, one bad 2D detection can pull the triangulated joint far from the true position because the learned depth-proposal head averages per-view candidates with fixed confidence.
* **Depth-proposal confidences are trained offline.** Once the checkpoint is frozen, the per-view weights do not adapt to the actual reprojection residuals of the current sample.
* **The residual/diffusion head fixes pose, not cameras.** v20/v21 refine pose/cameras during a single forward pass, but they do not iterate between triangulation and reprojection.

As a result, v25/v26 leave headroom on the table at low view counts, where a small amount of test-time self-correction can disproportionately help.

---

## 2. Proposed method

### 2.1 Core idea

At inference, treat the v25/v26 learned depth-proposal triangulation as an iterative re-weighted estimator. The predicted 3D pose is used to recompute per-view/joint reprojection residuals, which in turn update the confidences fed back into the triangulation head. This is an **IRLS-style loop inside a learned module**: the parameters of the model are frozen, only the per-sample confidences evolve.

### 2.2 New module

**File:** `motionflow_mv/fusion/test_time_self_evolution_v27.py`

```text
TestTimeSelfEvolutionV27(
    n_iters: int = 3,                       # number of self-evolution steps
    residual_thresh_mm: float = 0.5,      # early-stop if pose change < threshold
    sigma_reproj: float = 5.0,            # pixels, Huber/Cauchy scale for residuals
    use_temporal_smoothness: bool = True, # only when T > 1
    temporal_weight: float = 0.1,
    use_bone_length_regularizer: bool = True,  # reuse v22 KAP without gradient
)
```

### 2.3 Forward signature (drop-in for v25/v26)

```python
pred_3d_ref = tte(
    pred_3d_init,    # (B, T, J, 3)   output of v25/v26
    points_2d,       # (B, T, V, J, 2) detected 2D keypoints
    K,               # (B, T, V, 3, 3) intrinsics
    R,               # (B, T, V, 3, 3) rotations
    t,               # (B, T, V, 3)    translations
    view_mask,       # (B, T, V)       optional view mask
    confidence,      # (B, T, V, J)    optional initial confidences
)
```

**Outputs**

* `pred_3d_ref`: `(B, T, J, 3)` — refined 3D pose after self-evolution.

### 2.4 Algorithm

For step `k = 1..n_iters`:

1. **Reproject** current estimate:
   ```
   x_hat_vj = Pi(K_v, R_v, t_v, pred_3d_{k-1}[:, j])
   r_vj = x_hat_vj - points_2d[:, v, j]
   ```

2. **Update per-view confidence** with a Cauchy kernel:
   ```
   w_vj^k = confidence_vj / (1 + (||r_vj|| / sigma_reproj)^2)
   w_vj^k = w_vj^k * view_mask_v
   ```

3. **Re-run the learned v25 triangulation head** (or DLT fallback if the head is unavailable) with the updated weights to obtain `pred_3d_k`.  The v25 head is used in *inference-only* mode; its parameters are frozen.

4. **Apply a lightweight temporal smoothness prior** (when `T > 1`):
   ```
   pred_3d_k = pred_3d_k - temporal_weight * (pred_3d_k[:, 1:] - pred_3d_k[:, :-1])
   ```

5. **Apply a bone-length regularizer** by clamping joint positions to the v22 mean bone-length manifold (no gradients, pure post-processing).

6. **Early stop** if `mean(||pred_3d_k - pred_3d_{k-1}||) < residual_thresh_mm`.

### 2.5 Integration point

Insert the block in `motionflow_mv/fusion/omniview_fusion_v5.py` immediately after the existing v25/v26 hook (lines 787–802) and before the residual/diffusion head (line 805):

```python
# v25 / v26 geometry fusion already ran.
if self.use_test_time_self_evolution_v27 and self.test_time_self_evolution_v27 is not None:
    pred_3d_gn = self.test_time_self_evolution_v27(
        pred_3d_gn.view(B, T, J, 3),
        points_2d.view(B, T, V, J, 2),
        K_corrected.view(B, T, V, 3, 3),
        R.view(B, T, V, 3, 3),
        t.view(B, T, V, 3),
        view_mask=view_mask_flat.view(B, T, V),
        confidence=confidences.view(B, T, V, J),
    ).view(B * T, J, 3)
```

The block is **only active at `model.eval()`**; training always uses the single-shot v25/v26 output to keep the training graph simple.

---

## 3. Expected impact

| View count | Expected MPJPE change vs v25/v26 | Rationale |
|------------|------------------------------------|-----------|
| 2 views    | −8 % to −15 %                      | Single bad view dominates; re-weighting has large effect. |
| 4 views    | −5 % to −10 %                      | Moderate redundancy, still sensitive to outliers. |
| 8 views    | −3 % to −6 %                       | Redundancy reduces but re-weighting still cleans tails. |
| 14 views   | −2 % to −4 %                       | Baseline is already robust; gains are marginal. |

If v25 small reaches a `val_MPJPE` of ~19–20 mm on H36M, test-time self-evolution could drop it by **~0.8–1.5 mm** on the validation set, with larger relative gains on the 2-view benchmark. Runtime cost is a fixed 1–3 extra forward passes of the v25 triangulation head, which is small compared to the full model.

---

## 4. Implementation cost

* **Lines of code:** ~200–250 lines in `motionflow_mv/fusion/test_time_self_evolution_v27.py`, plus ~15 lines of hook wiring in `motionflow_mv/fusion/omniview_fusion_v5.py` (not implemented by this doc).
* **Training time:** None — the module is test-time only. Training does not change.
* **Data needs:** None new. The module reuses existing 2D keypoints and cameras.
* **Compute at inference:** 1–3 extra triangulation-head forward passes per sample; expected <20 % latency increase for the geometry block (the rest of the model is unchanged).

---

## 5. Risks / mitigation

| Risk | How it manifests | Mitigation |
|------|--------------------|------------|
| **Over-smoothing / divergence** | Iteration collapses to an implausible pose or oscillates. | Bound updates per step, use Cauchy re-weighting, and enforce early stopping. |
| **Test-time latency** | 3× geometry-head passes adds latency. | Default `n_iters=3`; can be reduced to `1` for real-time settings. |
| **Marginal gain at 14 views** | Little improvement because v25/v26 already stable. | Gate the loop by view count: skip if `V >= 10`. |
| **Bone-length regularizer over-constrains** | Unusual poses (sports, stretching) are pulled to mean bone lengths. | Make the bone-length regularizer a soft clamp with a small weight; fall back to zero if KAP is disabled. |
| **Dependency on v25 triangulation head** | If the head is only trained for single-shot use, it may not behave well in a loop. | First prototype uses the analytic DLT fallback inside the loop; later swap to the learned head once a smoke test shows stability. |

---

## 6. Minimal experiment plan

### 6.1 Flags / config names

Add to the YAML / model constructor:

```yaml
model:
  use_test_time_self_evolution_v27: true
  v27_n_iters: 3
  v27_residual_thresh_mm: 0.5
  v27_sigma_reproj: 5.0
  v27_use_temporal_smoothness: true
  v27_temporal_weight: 0.1
  v27_use_bone_length_regularizer: true
```

Constructor toggles in `OmniMultiViewFusionV5`:

```python
use_test_time_self_evolution_v27: bool = False,
v27_n_iters: int = 3,
v27_residual_thresh_mm: float = 0.5,
v27_sigma_reproj: float = 5.0,
v27_use_temporal_smoothness: bool = True,
v27_temporal_weight: float = 0.1,
v27_use_bone_length_regularizer: bool = True,
```

### 6.2 Smoke test

1. **Unit test:** `tests/test_test_time_self_evolution_v27.py`
   * Forward shape `(B, T, V, J, 3) -> (B, T, J, 3)`.
   * Identity when `n_iters=0`.
   * Early stopping triggers when `residual_thresh_mm` is large.
   * View masking is respected.
   * No gradient needed (module runs under `torch.no_grad()` at inference).

2. **Local smoke run on RTX 4090:**
   ```bash
   python experiments/benchmark_webbridge_h36m_test_smoke.py \
     --config configs/train_focal_calibration_smoke.yaml \
     --model.use_test_time_self_evolution_v27=true \
     --model.v27_n_iters=3
   ```
   Compare `val_MPJPE` and runtime to the same config with `use_test_time_self_evolution_v27=false`.

3. **Variable-view sweep:**
   Run the smoke config with `view_subset=[2,4,8,14]` and report per-view-count MPJPE. Gains should be largest at 2–4 views.

### 6.3 Go / no-go criteria

* **Go:** At least 3 % relative `val_MPJPE` improvement at 2 or 4 views with <30 % inference-time increase.
* **No-go:** Gains are <1 % or runtime increase >50 %; in that case, keep the code as an optional flag but do not invest in full A800 training.
