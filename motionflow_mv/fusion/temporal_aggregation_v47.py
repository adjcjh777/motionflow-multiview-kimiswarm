"""v47: Temporal Aggregation for sparse-view multi-view pose estimation.

This module adds a lightweight temporal refinement head on top of the per-frame
triangulated 3D poses produced by v46 sparse-view generalization.  It is a small
transformer encoder operating on ``(time, joint)`` tokens; at initialization the
residual path is zeroed so the module behaves as a no-op and the v46 per-frame
behaviour is preserved during warm-up.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TemporalAggregationV47(nn.Module):
    """Temporal aggregation head for refined 3-D pose trajectories.

    Parameters
    ----------
    n_joints:
        Number of skeleton joints.  Used for the joint positional embedding.
    d_model:
        Hidden dimension of the transformer encoder.
    n_heads:
        Number of self-attention heads.
    num_layers:
        Number of transformer encoder layers.
    temporal_window:
        If ``None``, tokens attend to every token in the clip.  If an odd
        integer, attention is restricted to ``window // 2`` frames on each
        side.  This is useful for low-latency / streaming settings.
    dropout:
        Dropout probability in the transformer encoder.
    residual_gate_init:
        Initial value of the scalar ``g`` that scales the residual refinement.
        ``0.0`` makes the temporal path a strict no-op at the start of training.
    use_view_count_conditioning:
        If ``True``, concatenate ``log(n_views_t)`` to each joint coordinate so
        the module can discount under-constrained frames.
    max_temporal_len:
        Maximum temporal length supported by the learned temporal positional
        embedding.  Longer clips are extended with zeros.
    """

    def __init__(
        self,
        n_joints: int = 17,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        temporal_window: Optional[int] = None,
        dropout: float = 0.1,
        residual_gate_init: float = 0.0,
        use_view_count_conditioning: bool = True,
        max_temporal_len: int = 256,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        if temporal_window is not None and temporal_window % 2 == 0:
            raise ValueError("temporal_window must be odd or None")

        self.n_joints = n_joints
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.temporal_window = temporal_window
        self.use_view_count_conditioning = use_view_count_conditioning
        self.max_temporal_len = max_temporal_len

        in_dim = 3 + (1 if use_view_count_conditioning else 0)

        self.input_proj = nn.Linear(in_dim, d_model)

        # Positional embeddings for time and joint index.
        self.temporal_pos = nn.Parameter(torch.randn(max_temporal_len, d_model) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(n_joints, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projects back to per-joint 3-D displacements.
        self.output_proj = nn.Linear(d_model, 3)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Learnable residual gate; identity behaviour when it is zero.
        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))

    def _build_local_window_mask(self, t: int, device: torch.device) -> torch.Tensor:
        """Return ``(T*J, T*J)`` additive mask that restricts temporal attention."""
        assert self.temporal_window is not None
        half = self.temporal_window // 2
        t_idx = torch.arange(t, device=device).unsqueeze(1)  # (T, 1)
        t_key = torch.arange(t, device=device).unsqueeze(0)  # (1, T)
        window = (t_idx - t_key).abs() <= half  # (T, T)

        # Each (time, joint) query attends to all joints in allowed frames.
        mask = window.unsqueeze(1).unsqueeze(3)  # (T, 1, T, 1)
        mask = mask.expand(t, self.n_joints, t, self.n_joints).reshape(t * self.n_joints, t * self.n_joints)
        return mask.float() * 0.0 + (1.0 - mask.float()) * -1e9

    def _normalize_view_mask(
        self,
        view_mask: Optional[torch.Tensor],
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a float ``(B, T, V)`` view mask, defaulting to all views valid."""
        if view_mask is None:
            return torch.ones(B, T, 1, device=device)

        if view_mask.dim() == 2:
            # (B, V) -> expand over T
            view_mask = view_mask.unsqueeze(1).expand(-1, T, -1)
        elif view_mask.dim() == 3:
            if view_mask.shape[1] == 1:
                view_mask = view_mask.expand(-1, T, -1)
        else:
            raise ValueError(f"view_mask must be (B, T, V) or (B, V), got shape {view_mask.shape}")

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
            ``(B, T, V)`` or ``(B, V)`` binary mask.  ``1`` / ``True`` means the
            view contributed to the corresponding frame.
        clip_mask:
            Optional ``(B, T)`` binary mask.  ``True`` marks a valid frame.

        Returns
        -------
        refined:
            ``(B, T, J, 3)`` refined poses.
        """
        if poses_3d.dim() != 4:
            raise ValueError(f"poses_3d must be (B, T, J, 3), got {poses_3d.dim()}D tensor")

        B, T, J, _ = poses_3d.shape
        if J != self.n_joints:
            raise ValueError(f"poses_3d has {J} joints but module was configured for {self.n_joints}")

        device = poses_3d.device
        x = poses_3d

        # Normalize view mask to (B, T, V).
        vm = self._normalize_view_mask(view_mask, B, T, device)
        V = vm.shape[-1]

        # Per-frame valid view count and its log.
        n_views_t = vm.sum(dim=-1).clamp(min=1.0)  # (B, T)
        log_n = torch.log(n_views_t)  # (B, T)

        # Combined per-frame validity: frame is valid if clip_mask says so and
        # there is at least one contributing view.
        frame_valid = n_views_t > 0.0
        if clip_mask is not None:
            clip_mask = clip_mask.to(device).float()
            if clip_mask.dim() == 1:
                clip_mask = clip_mask.unsqueeze(0).expand(B, -1)
            frame_valid = frame_valid & (clip_mask > 0.0)
        frame_valid = frame_valid.bool()  # (B, T)

        # Build input features per (time, joint) token.
        if self.use_view_count_conditioning:
            log_n_joints = log_n.unsqueeze(-1).unsqueeze(-1).expand(B, T, J, 1)  # (B, T, J, 1)
            features = torch.cat([x, log_n_joints], dim=-1)
        else:
            features = x

        # Flatten to token sequence.
        tokens = features.reshape(B, T * J, -1)  # (B, T*J, in_dim)
        tokens = self.input_proj(tokens)  # (B, T*J, d_model)

        # Add positional embeddings.
        if T > self.max_temporal_len:
            # Extend temporal positional embedding with zeros for unseen lengths.
            extra = T - self.max_temporal_len
            self.temporal_pos.data = torch.cat(
                [self.temporal_pos, torch.zeros(extra, self.d_model, device=device)],
                dim=0,
            )
            self.max_temporal_len = T

        t_pos = self.temporal_pos[:T].unsqueeze(1).expand(-1, J, -1).reshape(T * J, -1)
        j_pos = self.joint_pos.unsqueeze(0).expand(T, -1, -1).reshape(T * J, -1)
        tokens = tokens + t_pos[None, :, :] + j_pos[None, :, :]

        # Build masks.
        attn_mask: Optional[torch.Tensor] = None
        if self.temporal_window is not None:
            attn_mask = self._build_local_window_mask(T, device)

        # Key-padding mask per batch: invalid frames are ignored as keys.
        # Use a float additive mask so it matches the dtype of ``attn_mask`` and
        # works with the current PyTorch TransformerEncoder.
        key_padding_mask = torch.zeros(B, T * J, device=device)
        key_padding_mask = key_padding_mask.masked_fill(
            ~frame_valid.unsqueeze(-1).expand(B, T, J).reshape(B, T * J),
            -1e9,
        )

        # Transformer expects a float additive mask for the key padding.
        out = self.transformer(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        )  # (B, T*J, d_model)

        # Output residual.
        delta = self.output_proj(out).reshape(B, T, J, 3)

        # Do not modify frames that are invalid in the clip / have no views.
        delta = delta * frame_valid.unsqueeze(-1).unsqueeze(-1).float()

        refined = x + self.residual_gate * delta
        return refined


if __name__ == "__main__":
    B, T, J, V = 2, 9, 17, 4
    module = TemporalAggregationV47(n_joints=J, d_model=64, n_heads=4, num_layers=2)
    poses = torch.randn(B, T, J, 3)

    # Full views.
    view_mask = torch.ones(B, T, V)
    out = module(poses, view_mask=view_mask)
    assert out.shape == (B, T, J, 3)

    # Identity at init (residual gate zero, output projection zeroed).
    assert torch.allclose(out, poses, atol=1e-5)

    # Sparse / dropped views.
    view_mask[:, 1::2, -1] = 0.0
    out2 = module(poses, view_mask=view_mask)
    assert out2.shape == (B, T, J, 3)

    # Variable-length clip mask.
    clip_mask = torch.ones(B, T, dtype=torch.bool)
    clip_mask[:, -2:] = False
    out3 = module(poses, view_mask=view_mask, clip_mask=clip_mask)
    assert out3.shape == (B, T, J, 3)
    # Masked tail should equal input.
    assert torch.allclose(out3[:, -2:], poses[:, -2:], atol=1e-5)

    # Local window path.
    module_window = TemporalAggregationV47(
        n_joints=J,
        d_model=64,
        n_heads=4,
        num_layers=2,
        temporal_window=7,
    )
    out4 = module_window(poses, view_mask=view_mask)
    assert out4.shape == (B, T, J, 3)

    # Gradient sanity.
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in module.parameters())

    print("TemporalAggregationV47 CPU smoke test passed")
