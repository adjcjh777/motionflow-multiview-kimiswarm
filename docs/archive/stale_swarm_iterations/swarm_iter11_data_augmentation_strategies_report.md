# Iter11 Data-Augmentation Strategies for MotionFlow-MultiView

**Topic:** concrete, implementable data-augmentation improvements for the ICRA/CVPR 2027 submission roadmap.  
**Scope:** training pipeline for `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` and related ray-attention fusion models.  
**Date:** 2026-08-04.

---

## 1. Current state

All recent training scripts (`experiments/train_ray_attention_temporal_*.py`) share the same lightweight augmentation in `augment_clip`:

- Gaussian pixel noise: `x[..., :2] += N(0, 0.5²)`.
- Confidence dropout: 10% of `(T, V, J)` detections.
- 2D outliers: 2% replaced by a uniform `[-100, 100]` px offset.

This is applied to 13-frame clips from canonical `.npz` files.  The synthetic generator already produces H36M-matched rigs, but it is wired only to the older `RayAttentionFusionModelV3` transfer trainer.

The new all-in-one model adds uncertainty-weighted DLT, Gauss-Newton refinement, and a residual MLP head.  Each is sensitive to the *distribution* of 2D errors: detector noise, realistic residuals, and structured failures such as occlusion or missing views.  Basic pixel noise is no longer enough.

---

## 2. Proposed improvements

### A. Detector-realistic, joint-dependent 2D noise

Current augmentation uses a single `noise_std=0.5` px for every joint.  Replace it with joint-dependent covariance from training-set DLT residuals: per-joint `σ`, anisotropic `2×2` covariance for wrist/ankle blur, and confidence-aware scaling.

### B. Structured occlusion / view-dropout

Replace random dropout with:

- **Self-occlusion blobs:** zero contiguous joint blocks.
- **Camera-failure dropout:** drop an entire view for the whole clip.
- **Time-varying dropout:** zero a vertical image strip for a contiguous window.

### C. Synthetic data expansion with domain randomization

The existing generator matches H36M statistics tightly.  Add a domain-randomized pre-training mode:

- Random camera count `V ∈ {3, 4, 6, 8}`.
- Random baselines, heights, focal lengths, and principal points.
- Use AMASS motion clips instead of Brownian latent poses.

Pre-train on this diverse synthetic set, then fine-tune on real MPI-INF-3DHP / H36M.

### D. Temporal augmentation

Temporal models currently receive fixed 13-frame windows.  Add:

- **Speed perturbation:** sample at `stride ∈ {1, 2, 3}`.
- **Time reversal:** reverse the clip with 50% probability.
- **Window jitter:** randomize clip length between 9 and 17 frames.
- **Frame dropout:** blank 1–2 frames inside the clip.

### E. 3D-aware augmentation (mixup/cutmix in camera space)

Interpolate two clips in image space:

- `x_mixed = λ x_a + (1-λ) x_b`
- `y_mixed = λ y_a + (1-λ) y_b`

This forces the network to reason about intermediate poses.

### F. Hard negative mining via reprojection residual

After a warm-up epoch, re-sample clips with the largest reprojection residuals and apply stronger augmentation.

---

## 3. Metrics to track

| Metric | Target |
|--------|--------|
| MPI-INF-3DHP val MPJPE | < 11.17 mm |
| H36M S9/S11 MPJPE | < 5.74 mm |
| PA-MPJPE | track alongside MPJPE |
| Per-joint MPJPE | find limb vs. torso gains |
| View-dropout robustness | < 20% relative increase |
| Temporal jitter | reduce without smoothing penalty |
| Synthetic-to-real transfer gap | minimize Δ MPJPE |

---

## 4. Experiments to run

1. **Baseline:** train the new model with current `augment_clip` on MPI-INF-3DHP S1→S2.
2. **Joint-dependent noise:** replace uniform noise with calibrated per-joint covariance.
3. **Structured occlusion:** add view/camera dropout and occlusion blobs.
4. **Synthetic pre-training:** generate 15k domain-randomized frames with AMASS motion, pre-train, then fine-tune.
5. **Temporal augmentation:** add speed perturbation, reversal, and frame dropout.
6. **Combined best:** stack winning augmentations and train 30–50 epochs.

Use `configs/train_ray_attention_reproducible.yaml` as the reproducible template.

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Synthetic/real domain gap | Medium | Match camera intrinsics, use AMASS motion, validate transfer gap first |
| Strong augmentation underfits | Low/Medium | Start small, monitor train/val gap, use curriculum |
| Temporal augmentation breaks causality | Low | Reverse/stride only whole clips |
| New augmentations slow training | Medium | Vectorize in `augment_clip_v2` |
| Uncertainty head overfits to synthetic noise | Medium | Calibrate noise to real detector statistics |

---

## 6. Proposed code change

Replace the existing `augment_clip` in `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py` (and siblings) with the version below.

```python
import torch
import torch.nn.functional as F


def augment_clip_v2(
    x: torch.Tensor,
    noise_std: float = 0.5,
    joint_noise_scale: float = 1.0,
    dropout_rate: float = 0.1,
    view_dropout_rate: float = 0.1,
    occlusion_rate: float = 0.05,
    outlier_rate: float = 0.02,
    outlier_scale: float = 100.0,
    reverse_prob: float = 0.5,
    temporal_stride: int = 1,
) -> torch.Tensor:
    """Enhanced per-clip augmentation."""
    B, T, V, J, _ = x.shape

    # Temporal reversal and subsampling (kept as whole-clip operations).
    if reverse_prob > 0 and torch.rand(1).item() < reverse_prob:
        x = torch.flip(x, dims=[1])
    if temporal_stride > 1:
        x = x[:, ::temporal_stride]

    # Joint-dependent Gaussian noise: wrists/ankles get larger perturbations.
    per_joint_std = torch.ones(J, device=x.device, dtype=x.dtype)
    # Example heuristics; replace with empirical values per dataset.
    per_joint_std[[0, 1, 2, 3]] = noise_std  # torso/head
    per_joint_std[[4, 7, 11, 14]] = noise_std * joint_noise_scale * 1.5  # elbows/knee
    per_joint_std[[6, 7, 10, 11, 13, 14, 15, 16]] = noise_std * joint_noise_scale * 2.0  # wrist/ankle
    noise = torch.randn_like(x[..., :2]) * per_joint_std.view(1, 1, 1, J, 1)
    x[..., :2] = x[..., :2] + noise

    # Structured occlusion: contiguous joint blocks.
    if occlusion_rate > 0:
        occ = torch.rand(B, T, V, 1, device=x.device) < occlusion_rate
        x[..., 2] = x[..., 2] * (~occ).float()

    # Random view dropout (entire camera missing for the clip).
    if view_dropout_rate > 0:
        view_mask = torch.rand(B, 1, V, 1, device=x.device) > view_dropout_rate
        x[..., 2] = x[..., 2] * view_mask.float()

    # Confidence dropout.
    if dropout_rate > 0:
        mask = torch.rand(B, T, V, J, device=x.device) > dropout_rate
        x[..., 2] = x[..., 2] * mask.float()

    # 2D outliers.
    if outlier_rate > 0:
        outlier_mask = torch.rand(B, T, V, J, device=x.device) < outlier_rate
        outlier = (torch.rand(B, T, V, J, 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
        x[..., 2] = x[..., 2] * (~outlier_mask).float()

    return x
```

---

## 7. Summary and next step

The quickest win is **C + F**: generate a domain-randomized synthetic dataset with AMASS motion and train with the enhanced `augment_clip_v2`.  This addresses the limited real-data problem that produced the 47.54 mm fast run and gives the uncertainty/GN/residual heads a richer error distribution.  Next action: implement `augment_clip_v2`, add domain-randomized synthetic options, and run the first MPI-INF-3DHP S1→S2 ablation.
