"""v31: richer, geometry-aware camera-conditioned view embedding.

Replaces the flat ``K, R, t`` MLP with a structured embedding that (1) encodes
intrinsics and camera-center/optical-axis geometry, and (2) aggregates pairwise
view geometry (baseline, relative rotation, optical-axis alignment) through a
lightweight self-attention.  The final projection is zero-initialised so the block
is identity at init and the model falls back to the learned view positional
embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CameraConditionedViewEmbeddingV31(nn.Module):
    """Geometry-aware camera view embedding.

    Parameters
    ----------
    d:
        Output feature dimension.
    camera_hidden:
        Hidden dimension for the local camera-feature MLP.
    pairwise_hidden:
        Hidden dimension for the pairwise view-geometry branch.
    n_heads:
        Attention heads in the pairwise self-attention.
    dropout:
        Dropout in the pairwise self-attention.
    normalize_cameras:
        Whether to rescale intrinsics / translation before encoding.
    """

    def __init__(
        self,
        d: int,
        camera_hidden: int = 32,
        pairwise_hidden: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
        normalize_cameras: bool = True,
    ):
        super().__init__()
        self.d = d
        self.camera_hidden = camera_hidden
        self.pairwise_hidden = pairwise_hidden
        self.normalize_cameras = normalize_cameras

        # Local camera descriptor: fx, fy, cx, cy, camera center, optical axis,
        # relative center and distance to the scene centroid.
        local_dim = 4 + 3 + 3 + 3 + 1
        self.local_mlp = nn.Sequential(
            nn.Linear(local_dim, camera_hidden),
            nn.ReLU(),
            nn.LayerNorm(camera_hidden),
            nn.Linear(camera_hidden, camera_hidden),
            nn.ReLU(),
        )

        # Pairwise view-geometry descriptor: baseline, relative rotation angle,
        # optical-axis cosine.
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(3, pairwise_hidden),
            nn.ReLU(),
            nn.LayerNorm(pairwise_hidden),
            nn.Linear(pairwise_hidden, pairwise_hidden),
            nn.ReLU(),
        )
        self.pairwise_attn = nn.MultiheadAttention(
            embed_dim=pairwise_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Final fusion; zero-initialised for identity at init.
        self.out_proj = nn.Linear(camera_hidden + pairwise_hidden, d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _normalize(
        self, K: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale intrinsics and translation to a canonical magnitude."""
        K = K.clone()
        K[..., 0, 0] = K[..., 0, 0] / 1000.0
        K[..., 1, 1] = K[..., 1, 1] / 1000.0
        K[..., 0, 2] = K[..., 0, 2] / 320.0
        K[..., 1, 2] = K[..., 1, 2] / 240.0
        return K, t / 10.0

    def forward(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode camera parameters into per-view features.

        Args
        ----
        K:
            Intrinsics, ``(N, V, 3, 3)`` or ``(V, 3, 3)``.
        R:
            Rotation matrices, same shape as ``K``.
        t:
            Translation vectors, ``(N, V, 3)`` or ``(V, 3)``.
        view_mask:
            Optional binary mask of shape ``(N, V)``; ignored for now but kept
            for API compatibility.

        Returns
        -------
        Per-view embeddings of shape ``(N, V, d)``.
        """
        if K.dim() == 3:
            K = K.unsqueeze(0)
            R = R.unsqueeze(0)
            t = t.unsqueeze(0)

        if self.normalize_cameras:
            K, t = self._normalize(K, t)

        # Intrinsics.
        intrinsics = torch.stack(
            [K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]],
            dim=-1,
        )  # (N, V, 4)

        # Extrinsics.
        C = -torch.matmul(R.transpose(-2, -1), t.unsqueeze(-1)).squeeze(-1)
        optical_axis = R[..., 2, :]  # third world axis of each camera (z-axis)
        C_mean = C.mean(dim=1, keepdim=True)
        C_rel = C - C_mean
        dist = C_rel.norm(dim=-1, keepdim=True)

        local_feat = torch.cat([intrinsics, C, optical_axis, C_rel, dist], dim=-1)
        local_h = self.local_mlp(local_feat)

        # Pairwise view geometry.
        C_i = C.unsqueeze(2)
        C_j = C.unsqueeze(1)
        baseline = (C_i - C_j).norm(dim=-1, keepdim=True)

        R_rel = torch.matmul(R.transpose(-2, -1).unsqueeze(2), R.unsqueeze(1))
        trace = (
            R_rel[..., 0, 0]
            + R_rel[..., 1, 1]
            + R_rel[..., 2, 2]
        )
        angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)).unsqueeze(-1)

        optical_axis_i = optical_axis.unsqueeze(2)
        optical_axis_j = optical_axis.unsqueeze(1)
        dot = (optical_axis_i * optical_axis_j).sum(dim=-1, keepdim=True)

        pairwise_feat = torch.cat([baseline, angle, dot], dim=-1)
        pairwise_h = self.pairwise_mlp(pairwise_feat)

        N, V = pairwise_h.shape[0], pairwise_h.shape[1]
        pairwise_h_flat = pairwise_h.view(N * V, V, -1)

        # Optional padding mask for pairwise keys.
        key_padding_mask: torch.Tensor | None = None
        if view_mask is not None:
            # view_mask: (N, V); True/1 means valid view.
            key_padding_mask = view_mask.logical_not().view(N * V, V).bool()

        attn_out, _ = self.pairwise_attn(
            pairwise_h_flat, pairwise_h_flat, pairwise_h_flat,
            key_padding_mask=key_padding_mask,
        )
        # Pool the attended pairwise features back to one vector per view.
        pairwise_agg = attn_out.view(N, V, V, -1).mean(dim=2)

        fused = torch.cat([local_h, pairwise_agg], dim=-1)
        out = self.out_proj(fused)
        return out


if __name__ == "__main__":
    B, V, d = 2, 4, 64
    K = torch.randn(B, V, 3, 3) * 1000.0
    # Build valid rotation matrices.
    R = torch.linalg.qr(torch.randn(B, V, 3, 3))[0]
    t = torch.randn(B, V, 3)

    embed = CameraConditionedViewEmbeddingV31(d=d, camera_hidden=32, pairwise_hidden=32)
    out = embed(K, R, t)
    assert out.shape == (B, V, d)

    # Permutation equivariance.
    perm = torch.randperm(V)
    out_perm = embed(K[:, perm], R[:, perm], t[:, perm])
    assert torch.allclose(out[:, perm], out_perm, atol=1e-5, rtol=1e-4)

    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in embed.parameters())
    print("CameraConditionedViewEmbeddingV31 smoke test passed")
