"""Multi-scale cross-view spatial pyramid for calibrated multi-view pose fusion.

The pyramid augments per-frame features by operating cross-view attention at
multiple spatial (joint) scales.  At each scale the joint axis is downsampled,
a lightweight single-head cross-view attention block mixes information across
cameras, and the result is upsampled back to the original joint resolution.
The multi-scale outputs are fused with a learnable projection and a residual
connection.  This module is intentionally generic: it expects input of shape
(N, V, J, d) and returns a feature tensor of the same shape.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossViewBlock(nn.Module):
    """Lightweight cross-view attention + FFN block.

    Input / output shape: ``(B, V, d)``.
    """

    def __init__(self, d: int, n_views: int, n_heads: int = 1):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Linear(d * 2, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossViewSpatialPyramid(nn.Module):
    """Multi-scale cross-view spatial pyramid.

    For each downsample factor in ``scales`` the joint dimension is pooled to
    ``J // scale``, a cross-view attention block is applied, and the result is
    upsampled back to ``J``.  All scale branches are concatenated and projected
    back to ``d`` with a residual connection.

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of camera views.
    scales:
        Downsample factors for the joint axis.  ``1`` means full resolution.
        Default ``(1, 2, 4)``.
    n_heads:
        Number of attention heads in each cross-view block.  Default 1 to keep
        the module lightweight.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        scales: Sequence[int] = (1, 2, 4),
        n_heads: int = 1,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.scales = tuple(scales)

        if any(s < 1 for s in self.scales):
            raise ValueError("All scale factors must be >= 1")

        self.branches = nn.ModuleList(
            [_CrossViewBlock(d, n_views, n_heads=n_heads) for _ in self.scales]
        )

        # Learned fusion: concatenate multi-scale features and project back to d.
        self.fusion = nn.Sequential(
            nn.Linear(d * len(self.scales), d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape ``(N, V, J, d)``.

        Returns
        -------
        Tensor of shape ``(N, V, J, d)``.
        """
        N, V, J, d = x.shape
        x_in = x

        scale_features = []
        # Shared reshaping helper: (N, V, J, d) -> (N*J, V, d)
        for scale, branch in zip(self.scales, self.branches):
            if scale == 1:
                # Full joint resolution.
                x_s = x.permute(0, 2, 1, 3).reshape(N * J, V, d)
                x_s = branch(x_s)
                x_s = x_s.view(N, J, V, d).permute(0, 2, 1, 3)  # (N, V, J, d)
            else:
                target_j = max(1, J // scale)
                # (N, V, J, d) -> (N*V, d, J)
                x_perm = x.permute(0, 1, 3, 2).reshape(N * V, d, J)
                x_pooled = F.adaptive_avg_pool1d(x_perm, target_j)
                # (N*V, d, target_j) -> (N, target_j, V, d) -> (N*target_j, V, d)
                x_pooled = x_pooled.view(N, V, d, target_j).permute(0, 3, 1, 2)
                x_pooled = x_pooled.reshape(N * target_j, V, d)
                x_attended = branch(x_pooled)
                # (N*target_j, V, d) -> (N, target_j, V, d) -> (N*V, d, target_j)
                x_attended = x_attended.view(N, target_j, V, d).permute(0, 2, 3, 1)
                x_attended = x_attended.reshape(N * V, d, target_j)
                # Upsample back to J joints.
                x_upsampled = F.interpolate(
                    x_attended, size=J, mode="linear", align_corners=False
                )
                # (N*V, d, J) -> (N, V, J, d)
                x_s = x_upsampled.view(N, V, d, J).permute(0, 1, 3, 2)

            scale_features.append(x_s)

        # Fuse all scales.
        x_cat = torch.cat(scale_features, dim=-1)  # (N, V, J, d * S)
        x_out = self.fusion(x_cat)
        x_out = self.norm(x_out + x_in)
        return x_out




class AdaptiveScaleCrossViewSpatialPyramid(nn.Module):
    """Multi-scale cross-view spatial pyramid with learnable scale gating.

    Compared with ``CrossViewSpatialPyramid``, which concatenates all scale
    branches and projects them back to ``d`` channels, this variant learns a
    per-joint soft attention over scales.  The gate is conditioned on the
    average per-joint context across views, so the model can emphasize fine-
    resolution branches for precise end-effector joints and coarse-resolution
    branches for noisy/occluded torso joints.

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of camera views.
    scales:
        Downsample factors for the joint axis.  Default ``(1, 2, 4)``.
    n_heads:
        Number of attention heads in each cross-view block.  Default 1.
    gate_hidden:
        Hidden dimension of the scale-selection MLP.  Default ``d // 2``.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        scales: Sequence[int] = (1, 2, 4),
        n_heads: int = 1,
        gate_hidden: int | None = None,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.scales = tuple(scales)

        if any(s < 1 for s in self.scales):
            raise ValueError("All scale factors must be >= 1")

        self.branches = nn.ModuleList(
            [_CrossViewBlock(d, n_views, n_heads=n_heads) for _ in self.scales]
        )

        gate_hidden = gate_hidden or max(1, d // 2)
        self.scale_gate = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, len(self.scales)),
        )
        self.norm = nn.LayerNorm(d)

    def _downsample_branch(self, x: torch.Tensor, scale: int, branch: nn.Module) -> torch.Tensor:
        """Apply cross-view attention at ``J // scale`` resolution and upsample.

        Input / output shape: ``(N, V, J, d)``.
        """
        N, V, J, d = x.shape
        if scale == 1:
            x_s = x.permute(0, 2, 1, 3).reshape(N * J, V, d)
            x_s = branch(x_s)
            x_s = x_s.view(N, J, V, d).permute(0, 2, 1, 3)
            return x_s

        target_j = max(1, J // scale)
        x_perm = x.permute(0, 1, 3, 2).reshape(N * V, d, J)
        x_pooled = F.adaptive_avg_pool1d(x_perm, target_j)
        x_pooled = x_pooled.view(N, V, d, target_j).permute(0, 3, 1, 2)
        x_pooled = x_pooled.reshape(N * target_j, V, d)
        x_attended = branch(x_pooled)
        x_attended = x_attended.view(N, target_j, V, d).permute(0, 2, 3, 1)
        x_attended = x_attended.reshape(N * V, d, target_j)
        x_upsampled = F.interpolate(
            x_attended, size=J, mode="linear", align_corners=False
        )
        x_s = x_upsampled.view(N, V, d, J).permute(0, 1, 3, 2)
        return x_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape ``(N, V, J, d)``.

        Returns
        -------
        Tensor of shape ``(N, V, J, d)``.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input (N,V,J,d), got {x.shape}")

        N, V, J, d = x.shape
        x_in = x

        scale_features = []
        for scale, branch in zip(self.scales, self.branches):
            scale_features.append(self._downsample_branch(x, scale, branch))

        # Stack to (N, V, J, S, d).
        stack = torch.stack(scale_features, dim=3)

        # Per-joint scale attention, shared across views.
        context = x_in.mean(dim=1)  # (N, J, d)
        scale_logits = self.scale_gate(context)  # (N, J, S)
        scale_weights = F.softmax(scale_logits, dim=-1)  # (N, J, S)
        scale_weights = scale_weights.view(N, 1, J, len(self.scales), 1)

        x_out = (stack * scale_weights).sum(dim=3)  # (N, V, J, d)
        x_out = self.norm(x_out + x_in)
        return x_out
class CrossViewSpatialPyramidModel(nn.Module):
    """Tiny wrapper model used only for smoke tests.

    Consumes a 5-D multi-view clip ``(B, T, V, J, 3)`` and predicts 3D poses
    through a single linear layer so that the pyramid can be exercised in a
    standalone training loop.  This class is *not* part of the main pipeline.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 32,
        n_views: int = 4,
        scales: Sequence[int] = (1, 2, 4),
        n_heads: int = 1,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.obs_embed = nn.Linear(3, d)
        self.pyramid = CrossViewSpatialPyramid(
            d=d, n_views=n_views, scales=scales, n_heads=n_heads
        )
        self.head = nn.Linear(d, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict (B, T, J, 3) for a (B, T, V, J, 3) input."""
        B, T, V, J, _ = x.shape
        x_emb = self.obs_embed(x)  # (B, T, V, J, d)
        x_emb = x_emb.view(B * T, V, J, d)
        x_pyr = self.pyramid(x_emb)  # (B*T, V, J, d)
        x_pooled = x_pyr.mean(dim=1)  # (B*T, J, d)
        pred = self.head(x_pooled)  # (B*T, J, 3)
        return pred.view(B, T, J, 3)


def _make_toy_input(
    batch: int = 2, t: int = 5, v: int = 4, j: int = 17, d: int = 32
):
    """Build a deterministic toy input for the smoke test."""
    torch.manual_seed(0)
    return torch.randn(batch, t, v, j, 3)


if __name__ == "__main__":
    x = _make_toy_input()
    model = CrossViewSpatialPyramidModel(j=17, d=32, n_views=4, scales=(1, 2, 4))
    pred = model(x)
    assert pred.shape == (2, 5, 17, 3), pred.shape
    print("CrossViewSpatialPyramid smoke test passed.")
    print(f"  input shape:  {tuple(x.shape)}")
    print(f"  output shape: {tuple(pred.shape)}")
