# v51 Multi-View Diffusion Pose Refiner (MV-DPR)

**Focus area:** `multi_view_diffusion_refiner`  
**Proposed module file:** `motionflow_mv/fusion/multi_view_diffusion_refiner_v51.py`

## 1. Concept

Build a lightweight, few-step diffusion refiner that treats the fused 3-D pose as a noisy sample and learns to denoise it using multi-view geometric evidence. The module closes the v50 self-evolution loop by using the per-view reliability and per-joint uncertainty produced by the Self-Evolution Feedback Head as conditioning signals, rather than as simple weights. At inference it runs a small number of denoising steps (default 5), so it adds only modest latency while pushing sparse-view accuracy.

## 2. Architecture

The refiner takes the current 3-D pose `P_0 ∈ R^{J×3}` and corrupts it with Gaussian noise `ε` to `P_t`. A transformer denoiser predicts `ε_θ(P_t, t, C)` where `C` is a joint-wise conditioning vector built from:

- **Geometric residuals:** reprojection residuals `||x_v − Π_v(P_t)||_2` for each view `v`, weighted by the v50 SEFH per-view reliability.
- **Uncertainty scale:** `exp(−σ_j)` from the v50 per-joint log-variance.
- **Camera ray embedding:** the direction and distance from each joint to every camera center.
- **Domain token:** the v48 domain embedding, zeroed when not available.

The denoiser is a 2-layer transformer with 4 heads and a hidden size of 64, operating on `J` joint tokens. A residual skip connection adds the denoised correction to `P_0`; the skip is initialized so that at `t=0` the module is approximately identity. Noise scheduling follows a cosine diffusion schedule truncated to `T=10` training steps; inference uses DDIM with 5 steps.

## 3. New config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_multi_view_diffusion_refiner` | bool | `False` |
| `v51_mvdr_hidden` | int | `64` |
| `v51_mvdr_num_layers` | int | `2` |
| `v51_mvdr_num_heads` | int | `4` |
| `v51_mvdr_diffusion_steps` | int | `10` |
| `v51_mvdr_denoising_steps` | int | `5` |
| `v51_mvdr_noise_schedule` | str | `"cosine"` |
| `v51_mvdr_conditioning_dropout` | float | `0.1` |
| `v51_mvdr_geometry_loss_weight` | float | `0.1` |
| `v51_mvdr_loss_weight` | float | `1.0` |
| `v51_mvdr_identity_refiner` | bool | `True` |
| `v51_mvdr_max_views` | int | `8` |

## 4. Loss

During training, for `t ~ Uniform(1, T)`:

```
L_denoise = E_{P_0, t, ε} || ε_θ(P_t, t, C) − ε ||²
L_geom  = E [ Σ_v w_v · || x_v − Π_v(P_θ(P_t, t, C)) ||² ]
L_mvdr  = v51_mvdr_loss_weight · (L_denoise + v51_mvdr_geometry_loss_weight · L_geom)
```

`w_v` is the v50 SEFH reliability, and `P_θ(...)` is the predicted clean pose. The geometry loss is clamped by a Huber delta when training with large domain shifts.

## 5. Evaluation metric

Primary: `MPJPE@k` for `k = 2, 3, 4, full` and per-domain `MPJPE@2` on 3DPW actual. Secondary: inference latency (ms) and `Spearman(reliability, residual)` after refinement to confirm the self-evolution loop remains calibrated.

## 6. Expected MPJPE impact

- `MPJPE@2`: −2 to −4 mm by correcting joint positions that survive single-pass triangulation but violate multi-view consistency.
- `MPJPE@3`: −1 to −2 mm.
- `MPJPE@full`: ±0.5 mm, preserving the strong v46/v50 baseline.
- 3DPW actual `MPJPE@2`: −3 to −5 mm, because the denoiser can regularize toward plausible 3-D configurations under larger domain noise.

## 7. Main risk and mitigation

**Risk:** Iterative denoising can over-smooth distal joints or collapse to a mean pose, especially when only two views are available and the conditioning is weak.

**Mitigation:** (1) Identity-at-init residual connection so the baseline is preserved at `t=0`; (2) clamp noise magnitude to 30 % of the joint range during training; (3) cap inference to 5 DDIM steps; (4) condition on v50 reliability and reprojection residuals so the model never fully ignores geometric evidence; (5) disable the refiner for the first training epoch while the SEFH head stabilises.

## 8. Integration sketch

In `motionflow_mv/fusion/omniview_fusion_v5.py`, after the v50 SEFH pose/reliability output, optionally instantiate `MultiViewDiffusionRefinerV51`. It receives `P_0`, the 2-D keypoints, camera parameters, and the v50 reliability/uncertainty. It returns a refined pose `P_refined`, which becomes the final pose used for the supervised loss and for `eval_variable_views.py`.

## 9. Paper-story fit

This module turns the single-pass triangulation/fusion pipeline into a generative self-critic: the model proposes a pose, perturbs it, and learns to recover a better one from multi-view evidence. It directly supports the sparse-view and cross-domain narrative by exploiting the v50 uncertainty signal as conditioning, rather than as a post-hoc weight.
