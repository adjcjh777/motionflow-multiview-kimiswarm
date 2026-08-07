# v20: Diffusion-Based Pose Refinement Head

**Author:** swarm/design_v20_diffusion_refinement  
**Status:** Design / Prototype  
**Branch:** `swarm/v20_diffusion_refinement`

## 1. Motivation

The current v5 architecture ends with a deterministic residual MLP:

```python
residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)
delta = self.residual_mlp(residual_input)
pred_3d = pred_3d_gn + delta
```

This works well for small, structured corrections, but it cannot easily represent
multi-modal corrections (e.g. ambiguous occlusions) and is sensitive to the
initial triangulation error.  Diffusion models have shown strong results for
structured regression problems because they learn a distribution over possible
corrections rather than a single point estimate.

This proposal adds a **lightweight diffusion-based refinement head** that
treats the residual correction as a noise signal to be progressively denoised.
It is designed as a drop-in replacement for the deterministic residual MLP.

## 2. Module Overview

**File:** `motionflow_mv/fusion/diffusion_pose_refiner_v20.py`

`DiffusionPoseRefinerV20` is a small PyTorch module with the following
properties:

* **Inputs**
  * `pose_init`: `(B, T, J, 3)` or `(B, J, 3)` — coarse 3D pose from
    triangulation / Gauss-Newton.
  * `feat`: optional `(B, in_dim)` — pooled per-view features (e.g.
    `feat_pooled` from v5).
  * `train_targets`: optional `(B, T, J, 3)` or `(B, J, 3)` — ground-truth
    pose used to compute the diffusion loss during training.

* **Outputs**
  * Inference: `(B, T, J, 3)` refined pose.
  * Training: `(refined_pose, loss)` where `loss` is the MSE between predicted
    and true noise.

* **Key design choices**
  * Predicts the **residual** `delta = pose_target - pose_init`, not the full
    pose, so the initial triangulation remains a strong baseline.
  * Uses a small joint-level transformer (2 layers, 4 heads) as the denoiser.
  * Conditions on the noisy residual, the initial pose, and optional features.
  * Supports both linear and cosine beta schedules.
  * Inference runs a configurable number of DDPM steps (default 5) for fast
    deployment at test time.

## 3. Architecture Details

### 3.1 Diffusion Process

We define the residual target:

```
delta_0 = pose_target - pose_init
```

A forward diffusion process gradually adds Gaussian noise:

```
q(delta_t | delta_0) = N(sqrt(alpha_t) * delta_0,
                          (1 - alpha_t) * I)
```

The denoiser `epsilon_theta(delta_t, t, pose_init, feat)` predicts the noise
that was added, using a sinusoidal timestep embedding and the conditioning
signals.

### 3.2 Denoiser Network

```
input_proj:  (B, J, 3)          -> (B, J, H)
cond_proj:   (B, J, cond_dim)   -> (B, J, H)
time_mlp:    (B, time_dim)      -> (B, H) broadcast to (B, J, H)

blocks:      2 x JointAttentionBlock(H)
output_proj: (B, J, H)         -> (B, J, 3)
```

`cond_dim = 3 (noisy residual) + 3 (initial pose) + in_dim (optional features)`.

### 3.3 Inference

At inference, we start from pure noise and run DDPM sampling with a strided
schedule.  The number of inference steps is configurable (default 5, much fewer
than the 100 training steps) so the runtime cost is comparable to a small MLP.

## 4. Integration into v5

Replace the current residual MLP block in `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
# Before
residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)
delta = self.residual_mlp(residual_input)
pred_3d = pred_3d_gn + delta

# After (v20)
refiner = DiffusionPoseRefinerV20(
    j=J,
    in_dim=d,
    residual_hidden=128,
    num_diffusion_steps=100,
    num_inference_steps=5,
)
# During training, supply ground truth targets to compute diffusion loss.
refined, diff_loss = refiner(pred_3d_gn, feat=feat_pooled, train_targets=target_3d)
# Add diff_loss to the total training objective.
```

The change is non-breaking because `DiffusionPoseRefinerV20` is a standalone
module with its own parameters and loss.  Existing checkpoints that use the
MLP are unaffected.

## 5. Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_dim` | 64 | Dimension of the optional feature conditioning. |
| `residual_hidden` | 128 | Hidden dimension of the joint-level transformer. |
| `num_diffusion_steps` | 100 | Number of diffusion timesteps during training. |
| `num_inference_steps` | 5 | Number of DDPM sampling steps at inference. |
| `n_heads` | 4 | Attention heads in the joint-level transformer. |
| `beta_schedule` | `"linear"` | Noise schedule (`"linear"` or `"cosine"`). |

## 6. Test Coverage

Tests live in `tests/test_diffusion_pose_refiner_v20.py` and cover:

* Inference shape for spatiotemporal `(B, T, J, 3)` inputs.
* Training mode returns both a refined pose and a scalar loss.
* No-feature conditioning mode.
* Single-frame `(B, J, 3)` mode.
* Backward pass / gradient flow.
* Cosine schedule selection.

Run with:

```bash
source .venv/bin/activate
pytest tests/test_diffusion_pose_refiner_v20.py -v
```

## 7. Risks and Next Steps

* **Runtime:** 5 DDPM steps are still more expensive than one MLP forward.
    *Mitigation:* keep `num_inference_steps` small or distill the denoiser into
   a single-step student.
* **Training stability:** diffusion losses can be noisy early on.
    *Mitigation:* the module clamps the predicted residual to `[-1, 1]` meters
   and uses a small transformer.
* **Evaluation needed:** A/B test against the deterministic v5 residual MLP on
   H36M, MPI-INF-3DHP, and WebBridge validation sets.

## 8. References

1. Jonathan Ho, Ajay Jain, Pieter Abbeel. "Denoising Diffusion Probabilistic
   Models", NeurIPS 2020.
2. Robin Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion
   Models", CVPR 2022.
