"""Lightweight diffusion-based 3D pose refinement head (v20).

This module refines an initial 3D pose estimate by treating the residual
between the coarse pose and the true pose as a noise signal that can be
progressively denoised.  It is designed to be a drop-in replacement for
the deterministic residual MLP used in ``OmniMultiViewFusionV5``.

Design highlights
-----------------
* Small denoiser (MLP + joint-level attention) so training stays cheap.
* Predicts a residual correction rather than the full pose, preserving the
  initial triangulation as a strong baseline.
* Supports both training-time diffusion loss and inference-time fast
  deterministic sampling with a configurable number of steps.
* Conditions on pooled per-view features (optional) and the initial pose,
  making it compatible with the existing ``feat_pooled`` tensor in v5.

References
----------
* Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
* Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion
  Models", CVPR 2022.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep.

    Parameters
    ----------
    dim:
        Output dimension.  Must be even.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps -> (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device)
            * (torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1))
        )  # (half,)
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _JointAttentionBlock(nn.Module):
    """Lightweight self-attention block over joints."""

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, J, d)
        h = self.norm(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionPoseRefinerV20(nn.Module):
    """Lightweight diffusion refinement head for 3D human pose.

    The module receives a coarse 3D pose ``pose_init`` (e.g. the output of
    triangulation + Gauss-Newton in v5) and an optional feature vector
    ``feat``.  During training it learns to reverse a Gaussian diffusion
    process on the residual ``delta = pose_target - pose_init``.  During
    inference it runs a configurable number of denoising steps to produce a
    refined residual.

    Parameters
    ----------
    j:
        Number of joints.
    in_dim:
        Dimension of the conditioning feature vector (e.g. ``d`` from v5).
        If ``0`` or ``None``, no feature conditioning is used.
    residual_hidden:
        Hidden dimension of the denoiser.
    num_diffusion_steps:
        Total number of diffusion timesteps ``T``.
    num_inference_steps:
        Number of sampling steps used at inference (deterministic DDPM).
    n_heads:
        Number of attention heads in the joint-level transformer block.
    beta_schedule:
        Noise schedule, either ``"linear"`` or ``"cosine"``.

    Inputs
    ------
    pose_init: (B, J, 3) or (B, T, J, 3) coarse 3D pose.
    feat:      Optional (B, in_dim) conditioning features.
    train_targets: Optional (B, J, 3) or (B, T, J, 3) ground-truth pose used
                   during training.  If provided, the module returns both the
                   refined pose and the diffusion training loss.

    Outputs
    -------
    If ``train_targets`` is None (inference):
        refined_pose: (B, J, 3) or (B, T, J, 3)
    If ``train_targets`` is provided (training):
        (refined_pose, loss)
    """

    def __init__(
        self,
        j: int = 17,
        in_dim: Optional[int] = 64,
        residual_hidden: int = 128,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 5,
        n_heads: int = 4,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.j = j
        self.in_dim = in_dim or 0
        self.residual_hidden = residual_hidden
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps
        self.n_heads = n_heads

        # Diffusion schedule
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        elif beta_schedule == "cosine":
            steps = torch.arange(num_diffusion_steps + 1, dtype=torch.float32)
            s = 0.008
            f_t = torch.cos(((steps / num_diffusion_steps) + s) / (1 + s) * torch.pi / 2) ** 2
            alphas = f_t / f_t[0]
            betas = 1 - (alphas[1:] / alphas[:-1])
            betas = torch.clamp(betas, 1e-4, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Time embedding
        time_dim = 64
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Build conditioning vector dimension
        cond_dim = 3 + 3  # noisy residual + initial pose
        if self.in_dim > 0:
            cond_dim += self.in_dim

        # Input projection
        self.input_proj = nn.Linear(3, residual_hidden)
        self.cond_proj = nn.Linear(cond_dim, residual_hidden)

        # Joint-level attention blocks
        self.blocks = nn.ModuleList(
            [
                _JointAttentionBlock(residual_hidden, n_heads=n_heads)
                for _ in range(2)
            ]
        )

        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, residual_hidden),
            nn.GELU(),
            nn.Linear(residual_hidden, residual_hidden),
        )

        # Output head predicts the noise (residual correction)
        self.output_proj = nn.Linear(residual_hidden, 3)

        # Limit residual magnitude for stability at the start of training.
        self.max_residual = 1.0

    def _add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """q(x_t | x_0)."""
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise

    def _denoise_step(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise for a single batch of noisy residuals."""
        # x_t: (B, J, 3), cond: (B, J, cond_dim)
        h = self.input_proj(x_t)  # (B, J, H)
        c = self.cond_proj(cond)  # (B, J, H)
        h = h + c

        # Add timestep conditioning as a bias (broadcast over joints)
        t_emb = self.time_embed(t)  # (B, time_dim)
        t_bias = self.time_mlp(t_emb)[:, None, :]  # (B, 1, H)
        h = h + t_bias

        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)  # (B, J, 3)

    def forward(
        self,
        pose_init: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        train_targets: Optional[torch.Tensor] = None,
    ):
        """Forward pass.  See class docstring for shapes."""
        orig_shape = pose_init.shape
        if pose_init.dim() == 4:
            # (B, T, J, 3) -> (B*T, J, 3)
            B, T, J, _ = pose_init.shape
            pose_init_flat = pose_init.reshape(B * T, J, 3)
            if feat is not None:
                feat_flat = feat.reshape(B * T, -1) if feat.dim() == 2 else feat.reshape(B * T, -1)
            else:
                feat_flat = None
            if train_targets is not None:
                train_targets_flat = train_targets.reshape(B * T, J, 3)
            else:
                train_targets_flat = None
        else:
            pose_init_flat = pose_init
            feat_flat = feat
            train_targets_flat = train_targets
            J = pose_init_flat.shape[1]

        if train_targets_flat is not None:
            # Training: predict noise and return diffusion loss + a deterministic
            # refinement from a randomly sampled timestep.
            residual_target = train_targets_flat - pose_init_flat
            residual_target = torch.clamp(residual_target, -self.max_residual, self.max_residual)
            B_local = pose_init_flat.shape[0]
            t = torch.randint(0, self.num_diffusion_steps, (B_local,), device=pose_init.device)
            noise = torch.randn_like(residual_target)
            noisy_residual = self._add_noise(residual_target, t, noise)

            cond = self._build_conditioning(noisy_residual, pose_init_flat, feat_flat)
            predicted_noise = self._denoise_step(noisy_residual, t, cond)
            loss = F.mse_loss(predicted_noise, noise)

            # Also return a deterministic refinement using the mean predictor.
            refined = self._sample(pose_init_flat, feat_flat)
            if pose_init.dim() == 4:
                refined = refined.reshape(orig_shape)
            return refined, loss

        # Inference
        refined = self._sample(pose_init_flat, feat_flat)
        if pose_init.dim() == 4:
            refined = refined.reshape(orig_shape)
        return refined

    def _build_conditioning(
        self,
        noisy_residual: torch.Tensor,
        pose_init: torch.Tensor,
        feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B_local, J_local, _ = noisy_residual.shape
        parts = [noisy_residual, pose_init]
        if feat is not None and self.in_dim > 0:
            feat_expanded = feat[:, None, :].expand(-1, J_local, -1)
            parts.append(feat_expanded)
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def _sample(
        self,
        pose_init: torch.Tensor,
        feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Deterministic DDPM sampling."""
        B_local, J_local, _ = pose_init.shape
        device = pose_init.device
        x = torch.randn_like(pose_init)

        # Use a strided schedule for fast inference.
        total_steps = self.num_diffusion_steps
        inference_steps = max(1, min(self.num_inference_steps, total_steps))
        if inference_steps == total_steps:
            timesteps = torch.arange(total_steps - 1, -1, -1, device=device)
        else:
            timesteps = torch.linspace(total_steps - 1, 0, inference_steps, device=device).long()

        for t_val in timesteps:
            t = torch.full((B_local,), t_val.item(), device=device, dtype=torch.long)
            cond = self._build_conditioning(x, pose_init, feat)
            predicted_noise = self._denoise_step(x, t, cond)

            alpha_t = self.alphas[t_val]
            beta_t = self.betas[t_val]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_val]
            sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_val]

            # x_0 prediction
            x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t

            if t_val > 0:
                sqrt_alpha_cumprod_prev = self.sqrt_alphas_cumprod[t_val - 1]
                # DDPM posterior mean
                coef1 = sqrt_alpha_cumprod_prev * beta_t / (1.0 - self.alphas_cumprod[t_val - 1])
                coef2 = sqrt_alpha_cumprod_t * (1.0 - self.alphas_cumprod[t_val - 1]) / (1.0 - self.alphas_cumprod[t_val]) * torch.sqrt(alpha_t)
                mean = coef1 * x0_pred + coef2 * x
                # Posterior variance
                var = (1.0 - self.alphas_cumprod[t_val - 1]) / (1.0 - self.alphas_cumprod[t_val]) * beta_t
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = x0_pred

        # Clamp final residual and add to initial pose.
        x = torch.clamp(x, -self.max_residual, self.max_residual)
        return pose_init + x


if __name__ == "__main__":
    # CPU smoke test
    torch.manual_seed(0)
    B, T, J = 2, 7, 17
    pose = torch.randn(B, T, J, 3)
    feat = torch.randn(B * T, 64)

    refiner = DiffusionPoseRefinerV20(
        j=J,
        in_dim=64,
        residual_hidden=32,
        num_diffusion_steps=10,
        num_inference_steps=3,
    )

    # Training mode
    targets = torch.randn(B, T, J, 3)
    refined, loss = refiner(pose, feat=feat, train_targets=targets)
    assert refined.shape == (B, T, J, 3)
    assert loss.numel() == 1

    # Inference mode
    refined2 = refiner(pose, feat=feat)
    assert refined2.shape == (B, T, J, 3)

    print("DiffusionPoseRefinerV20 CPU smoke test passed")
