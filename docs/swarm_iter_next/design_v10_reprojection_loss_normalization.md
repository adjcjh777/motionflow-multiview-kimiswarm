# v10 Design Proposal: Reprojection-Loss Normalization and Stability

## 1. Problem in v7 / v8 / v9

`experiments/train_omniview_fusion_v5_webbridge_multi.py` adds a 2-D reprojection term via `_reprojection_loss`:

```python
diff = x_pred - points_2d
return (diff ** 2 * mask).sum() / (mask.sum() + 1e-8)
```

This is raw **pixel-squared** error. For a ~1 MPx frame, a 10 px residual already contributes `10^2 = 100` per point; with `V*J ≈ 4×17 = 68` points, the reprojection loss is on the order of `10^3–10^4`. The primary 3-D MSE loss (`F.mse_loss(pred_3d, y)`) is in metric units (WebBridge canonical is meters), so a 10 mm error contributes `1e-4–1e-6`. Even with `--reproj_loss_weight 0.01`, the reprojection gradient dominates and destabilizes training.

v7 trains stably because only the 3-D MSE + epipolar terms drive it. v8 added robust covariance-aware DLT reweighting, which is internally stable after clamping. v9 added

* 2-D reprojection loss (unnormalized pixel MSE)
* PA (Procrustes-aligned) loss
* Monotonic multi-view ranking loss

and immediately blew up to ~3000+ loss at step 50-100. The likely root cause is **loss-scale mismatch and unnormalized, non-robust reprojection error**, amplified by the extra forward pass the monotonic loss requires.

## 2. Specific, Implementable Change (v10)

Make the 2-D reprojection loss **scale-invariant, robust, and scheduled**.

### 2.1 Normalize by focal length

Replace raw pixel error with error normalized by the per-view mean focal length:

```python
f = (K[..., 0, 0] + K[..., 1, 1]) / 2  # (B, T, V) or (B, V)
diff_norm = diff / f.unsqueeze(-1)       # (B, T, V, J, 2)
```

This maps pixel residuals to approximately **sensor-plane angle units**, putting the reprojection loss on the same order of magnitude as the 3-D MSE loss (`1e-4–1e-2`).

### 2.2 Robust Charbonnier loss

Replace squared error with the Charbonnier (pseudo-Huber) loss to suppress outliers from occluded/noisy views:

```python
rho = torch.sqrt(diff_norm ** 2 + eps ** 2)   # eps ≈ 1e-4 in normalized units
loss = (rho * mask).sum() / (mask.sum() + 1e-8)
```

This prevents a single bad view from exploding the gradient while still giving a clean quadratic near zero.

### 2.3 Confidence-aware masking

Weight each residual by the input confidence and predicted visibility so that dropped/occluded views do not participate:

```python
w = confidences * visibility                # already available in compute_loss
loss = (rho * w * mask).sum() / (w.sum() * mask.sum() + 1e-8)
```

### 2.4 Warm-up / schedule the reprojection weight

Start `reproj_loss_weight` at 0 and linearly ramp it to the target value over the first `N` steps (e.g. `N = 1000`). This lets the 3-D pose stabilize before the 2-D term pulls it. The existing `args.reproj_loss_weight` becomes the target, not the initial value.

### 2.5 Defer / simplify the monotonic loss

The monotonic loss in v9 does a second full forward pass and compares subset vs. full error. It is expensive and, in the presence of unnormalized reprojection error, can amplify instability. For v10:

* **Default:** disable monotonic loss (`--monotonic_loss_weight 0`)
* **Optional follow-up:** re-enable it only after the reprojection loss has been validated as stable.

### 2.6 PA loss sanity check

PA loss uses Procrustes alignment, which is scale-sensitive in the first epochs. Keep it enabled only if its weight is already small; otherwise schedule it alongside the reprojection loss.

### Files to touch

| File | Change |
|------|--------|
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Rewrite `_reprojection_loss` with focal normalization, Charbonnier loss, and confidence weighting. Add reprojection weight warm-up inside `build_compute_loss`. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | No change required; reprojection is computed from outputs already returned. |
| `motionflow_mv/fusion/attention_entropy_loss.py` | No change required. |

Pseudo-code for the new `_reprojection_loss`:

```python
def _reprojection_loss(pred_3d, points_2d, K, R, t, view_mask, confidences, visibility, eps=1e-4):
    # pred_3d: (B, T, J, 3)
    # points_2d: (B, T, V, J, 2)
    # K: (B, T, V, 3, 3)
    # view_mask: (B, T, V)
    # confidences, visibility: (B, T, V, J)
    uv_pred = project_points_3d_to_2d(pred_3d, K, R, t)   # (B, T, V, J, 2)
    diff = uv_pred - points_2d
    f = (K[..., 0, 0] + K[..., 1, 1]) / 2                # (B, T, V)
    diff_norm = diff / f.unsqueeze(-1).unsqueeze(-1)       # normalize by focal length
    rho = torch.sqrt(diff_norm ** 2 + eps ** 2)
    w = confidences * visibility
    mask = view_mask.unsqueeze(-1).unsqueeze(-1)
    numerator = (rho * w * mask).sum()
    denominator = (w * mask).sum() + 1e-8
    return numerator / denominator
```

And in `build_compute_loss`, warm up the reprojection weight:

```python
step = trainer.global_step if hasattr(trainer, "global_step") else 0
warmup_steps = args.reproj_loss_warmup_steps  # e.g. 1000
alpha = min(1.0, step / max(1, warmup_steps))
reproj_weight = args.reproj_loss_weight * alpha
```

## 3. Validation Plan

### 3.1 Smoke test (minutes)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_full_precision_dlt \
  --use_robust_dlt_reweight \
  --reproj_loss_weight 0.1 \
  --reproj_loss_warmup_steps 100
```

Expect: completes 1 epoch, no NaN/Inf, `reproj_loss` < 1.0 (normalized units).

### 3.2 Small fast run (1–2 hours on A800-D)

Use a single WebBridge split (e.g. H36M only, 4 views) and compare three runs:

| Run | Configuration |
|-----|---------------|
| A (baseline v8) | `--use_full_precision_dlt --use_robust_dlt_reweight --reproj_loss_weight 0` |
| B (v9 repro) | `--reproj_loss_weight 0.1 --pa_loss_weight 0.1 --monotonic_loss_weight 0.1` with old unnormalized loss |
| C (v10) | `--reproj_loss_weight 0.1 --reproj_loss_warmup_steps 1000 --pa_loss_weight 0.05 --monotonic_loss_weight 0` with new normalized robust loss |

Track per-step: total loss, `mpjpe`, `reproj_loss`, `pa_loss`, gradient norm. Success for v10:

* total loss stays below ~100 after step 100 (vs. v9's ~3000)
* `reproj_loss` and 3-D MSE are within one order of magnitude
* val MPJPE is within 2 mm of v8 after 2 epochs

### 3.3 Full run (A800-D, WebBridge H36M+MPI mixed, variable views)

If the small run is stable, run the standard 30-epoch mixed-dataset schedule:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight \
  --use_domain_embedding \
  --use_variable_view_training \
  --variable_view_min_views 2 --variable_view_max_views 4 \
  --reproj_loss_weight 0.05 \
  --reproj_loss_warmup_steps 1000 \
  --pa_loss_weight 0.05 \
  --monotonic_loss_weight 0.0
```

Primary metrics: MPJPE on H36M test and MPI-INF-3DHP test, training loss stability, worst-case MPJPE under variable views.

## 4. Expected Impact

| Aspect | Expected Change |
|--------|-----------------|
| **Training stability** | Total loss no longer spikes; reprojection and 3-D losses live on comparable scales. |
| **MPJPE** | At least no regression vs. v8; likely small improvement because the 3-D pose is explicitly tied to 2-D observations. |
| **Robustness** | Charbonnier loss + confidence weighting makes the model less sensitive to one noisy view or occlusion. |
| **Variable-view training** | Stable because the loss is normalized by focal length, so different camera rigs (H36M vs. MPI) contribute similarly. |
| **Training speed** | Slightly faster per step than v9 because the monotonic loss (second forward pass) is disabled by default. |

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Focal-length normalization still leaves dataset scale differences** | Also clip `f` to a minimum based on image diagonal; or additionally normalize by `max(K.shape image_width, image_height)`. |
| **Charbonnier `eps` too large, suppressing useful signal** | Start with `eps = 1e-4` (normalized units, ~0.1 px for f≈1000); tune on the small run. |
| **Warm-up delays the benefit of reprojection supervision** | Keep warm-up short (≤ 1000 steps); the target weight can be reached within the first epoch. |
| **PA loss still conflicts if unscheduled** | Either keep `pa_loss_weight` small (0.05) or warm it up together with the reprojection loss. |
| **Removing monotonic loss loses a useful regularizer** | Re-enable it only after v10 is validated; it is not the cause of the immediate blow-up. |

## 6. Next Step Recommendation

Proceed with the minimal change in §2: **rewrite `_reprojection_loss` with focal-length normalization + Charbonnier loss + confidence weighting, add a 1000-step warm-up, and disable the monotonic loss**. Run the smoke test and the small 1–2 epoch comparison before committing to the full 30-epoch run.
