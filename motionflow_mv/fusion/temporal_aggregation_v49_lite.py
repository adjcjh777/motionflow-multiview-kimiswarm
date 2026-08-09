"""v49-Lite: lightweight causal temporal aggregation for RTX 4090 smoke/full runs.

Replaces the v47 transformer encoder with a small stack of causal depthwise
separable Conv1D blocks.  The module is designed to be cheap at training and
inference time while still providing temporal smoothing and a per-frame
uncertainty signal.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CausalDepthwiseSeparableConv1d(nn.Module):
    """Causal depthwise-separable Conv1D block with residual and GLU gating."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for causal convolution")
        self.padding = kernel_size - 1
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=0,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, C)
        Returns:
            (B, T, C)
        """
        B, T, C = x.shape
        # Causal padding on the left.
        x_padded = torch.nn.functional.pad(x.transpose(1, 2), (self.padding, 0))
        out = self.depthwise(x_padded)  # (B, C, T)
        out = self.pointwise(out).transpose(1, 2)  # (B, T, 2*C)
        out, gate = out.chunk(2, dim=-1)
        out = out * torch.sigmoid(gate)
        out = self.dropout(out)
        # Residual connection with same shape.
        out = self.norm(x + out)
        return out


class TemporalAggregationV49Lite(nn.Module):
    """Lightweight causal temporal refinement head.

    Parameters
    ----------
    n_joints:
        Number of skeleton joints.
    d_model:
        Hidden dimension for the joint-coordinate embedding.
    num_layers:
        Number of causal Conv1D blocks.
    kernel_size:
        Kernel size for causal Conv1D (odd).
    dropout:
        Dropout probability.
    residual_gate_init:
        Initial scalar residual gate (clamped to [0,1]).
    use_view_count_conditioning:
        If ``True``, concatenate ``log(n_views_t)`` to each joint coordinate.
    """

    def __init__(
        self,
        n_joints: int = 17,
        d_model: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        residual_gate_init: float = 0.0,
        use_view_count_conditioning: bool = True,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.d_model = d_model
        self.use_view_count_conditioning = use_view_count_conditioning

        in_dim = 3 + (1 if use_view_count_conditioning else 0)
        self.input_proj = nn.Linear(in_dim, d_model)

        self.blocks = nn.ModuleList(
            [CausalDepthwiseSeparableConv1d(d_model, kernel_size=kernel_size, dropout=dropout)
             for _ in range(num_layers)]
        )

        self.output_proj = nn.Linear(d_model, 3)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))

    def _normalize_view_mask(
        self,
        view_mask: Optional[torch.Tensor],
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        if view_mask is None:
            return torch.ones(B, T, 1, device=device)
        if view_mask.dim() == 2:
            view_mask = view_mask.unsqueeze(1).expand(-1, T, -1)
        elif view_mask.dim() == 3:
            if view_mask.shape[1] == 1:
                view_mask = view_mask.expand(-1, T, -1)
        else:
            raise ValueError(f"view_mask must be (B, T, V) or (B, V), got {view_mask.shape}")
        return view_mask.float().to(device)

    def forward(
        self,
        poses_3d: torch.Tensor,
        view_mask: torch.Tensor,
        clip_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return temporally refined 3-D poses.

        Args
        ----
        poses_3d:
            ``(B, T, J, 3)`` per-frame triangulated poses.
        view_mask:
            ``(B, T, V)`` or ``(B, V)`` binary mask.
        clip_mask:
            Optional ``(B, T)`` binary mask.

        Returns
        -------
        refined:
            ``(B, T, J, 3)`` refined poses.
        """
        if poses_3d.dim() != 4:
            raise ValueError(f"poses_3d must be (B, T, J, 3), got {poses_3d.dim()}D tensor")
        B, T, J, _ = poses_3d.shape
        if J != self.n_joints:
            raise ValueError(f"poses_3d has {J} joints but module expects {self.n_joints}")

        device = poses_3d.device
        x = poses_3d

        vm = self._normalize_view_mask(view_mask, B, T, device)
        n_views_t = vm.sum(dim=-1).clamp(min=1.0)  # (B, T)
        frame_valid = n_views_t > 0.0
        if clip_mask is not None:
            clip_mask = clip_mask.to(device).float()
            if clip_mask.dim() == 1:
                clip_mask = clip_mask.unsqueeze(0).expand(B, -1)
            frame_valid = frame_valid & (clip_mask > 0.0)
        frame_valid = frame_valid.bool()  # (B, T)

        if self.use_view_count_conditioning:
            log_n = torch.log(n_views_t).unsqueeze(-1).unsqueeze(-1).expand(B, T, J, 1)
            features = torch.cat([x, log_n], dim=-1)
        else:
            features = x

        # (B, T, J, in_dim) -> (B, T, J, d_model)
        tokens = self.input_proj(features)

        # Apply temporal Conv1D over time dimension for each joint/channel independently.
        # Reshape to (B*J, T, d_model), process, then reshape back.
        bj_tokens = tokens.permute(0, 2, 1, 3).reshape(B * J, T, self.d_model)
        for block in self.blocks:
            bj_tokens = block(bj_tokens)
        tokens = bj_tokens.reshape(B, J, T, self.d_model).permute(0, 2, 1, 3)

        delta = self.output_proj(tokens)  # (B, T, J, 3)
        delta = delta * frame_valid.unsqueeze(-1).unsqueeze(-1).float()

        gate = self.residual_gate.clamp(0.0, 1.0)
        refined = x + gate * delta
        return refined


if __name__ == "__main__":
    B, T, J, V = 2, 9, 17, 4
    module = TemporalAggregationV49Lite(n_joints=J, d_model=32, num_layers=2)
    poses = torch.randn(B, T, J, 3)
    view_mask = torch.ones(B, T, V)
    out = module(poses, view_mask=view_mask)
    assert out.shape == (B, T, J, 3)
    # Identity at init.
    assert torch.allclose(out, poses, atol=1e-5)

    # Sparse views.
    view_mask[:, 1::2, -1] = 0.0
    out2 = module(poses, view_mask=view_mask)
    assert out2.shape == (B, T, J, 3)

    # Clip mask.
    clip_mask = torch.ones(B, T, dtype=torch.bool)
    clip_mask[:, -2:] = False
    out3 = module(poses, view_mask=view_mask, clip_mask=clip_mask)
    assert torch.allclose(out3[:, -2:], poses[:, -2:], atol=1e-5)

    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in module.parameters())
    print("TemporalAggregationV49Lite CPU smoke test passed")
