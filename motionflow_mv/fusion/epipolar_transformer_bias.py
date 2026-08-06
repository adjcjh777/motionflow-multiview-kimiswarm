"""Epipolar-geometry-aware relative-position bias for spatio-temporal attention.

This module turns calibrated multi-view geometry into a differentiable relative-
position bias that is injected into the self-attention scores of the cross-view
spatio-temporal transformer.  Unlike ``epipolar_attention_bias`` (v1), which
only biases the final per-view weight head, the bias here shapes *feature*
fusion itself, making the transformer geometry-aware end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .epipolar_attention_bias import compute_epipolar_distance


def _aggregate_pairwise_epipolar_distance(K, R, t, points_2d):
    """Compute a symmetric, per-joint epipolar distance for every view pair.

    Args
    ----
    K: ``(N, V, 3, 3)`` intrinsic matrices.
    R: ``(N, V, 3, 3)`` rotation matrices.
    t: ``(N, V, 3)`` translation vectors.
    points_2d: ``(N, V, J, 2)`` image points.

    Returns
    -------
    dist: ``(N, V, V, J)`` symmetric epipolar-line distance between each pair of
    views for each joint.
    """
    # ``compute_epipolar_distance`` returns the distance from the destination
    # view point to the epipolar line induced by the source view point.
    dist = compute_epipolar_distance(K, R, t, points_2d)  # (N, V, V, J)
    # Symmetrize the directed distance to obtain a pair-wise consistency score.
    dist_sym = 0.5 * (dist + dist.transpose(1, 2))  # (N, V, V, J)
    return dist_sym


def compute_per_frame_epipolar_bias(K, R, t, points_2d, temperature=10.0):
    """Compute per-frame epipolar bias blocks from intra-frame epipolar geometry.

    Args
    ----
    K: ``(B*T, V, 3, 3)`` intrinsic matrices.
    R: ``(B*T, V, 3, 3)`` rotation matrices.
    t: ``(B*T, V, 3)`` translation vectors.
    points_2d: ``(B*T, V, J, 2)`` image points.
    temperature: positive scalar scaling the distance.

    Returns
    -------
    bias: ``(B*T, V, V)`` additive bias for the intra-frame (V, V) attention
    block.  Lower (more negative) values indicate larger epipolar inconsistency.
    """
    dist = _aggregate_pairwise_epipolar_distance(K, R, t, points_2d)  # (N, V, V, J)
    per_frame_bias = -dist.mean(dim=-1) / temperature  # (N, V, V)
    return per_frame_bias


def build_temporal_bias_from_frames(per_frame_bias, n_heads=4, n_joints=1):
    """Build an 3-D attention mask for ``MultiheadAttention`` from per-frame epipolar bias.

    Args
    ----
    per_frame_bias: ``(B, T, V, V)`` additive biases, one ``(V, V)`` block per frame.
    n_heads: number of attention heads to replicate over.
    n_joints: number of joints in the ST transformer batch (the transformer
        processes one joint per sample, so the bias is repeated for each joint).

    Returns
    -------
    attn_bias: ``(B*n_joints*n_heads, T*V, T*V)`` tensor ready for
    ``MultiheadAttention.forward(attn_mask=...)``.
    """
    B, T, V, _ = per_frame_bias.shape
    bias = torch.zeros(B, T * V, T * V, device=per_frame_bias.device, dtype=per_frame_bias.dtype)
    for t in range(T):
        bias[:, t * V:(t + 1) * V, t * V:(t + 1) * V] = per_frame_bias[:, t]
    # Repeat for each joint, then for each head.
    bias = bias.unsqueeze(1).repeat(1, n_joints, 1, 1)  # (B, J, L, S)
    bias = bias.reshape(B * n_joints, 1, T * V, T * V)
    bias = bias.repeat(1, n_heads, 1, 1)  # (B*J, n_heads, L, S)
    bias = bias.reshape(B * n_joints * n_heads, T * V, T * V)
    return bias


class EpipolarBiasedTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer that accepts an epipolar attention bias.

    Mirrors ``nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
    dim_feedforward=d*2, batch_first=True, norm_first=True)`` but accepts an
    optional ``epipolar_bias`` tensor that is added to raw attention scores.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=True,
                 norm_first=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
            **factory_kwargs,
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, epipolar_bias=None):
        """Args: src (N, L, d); epipolar_bias (N*n_heads, L, L) or (L, L)."""
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), epipolar_bias)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, epipolar_bias))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, epipolar_bias):
        x2, _ = self.self_attn(
            x, x, x,
            attn_mask=epipolar_bias,
            need_weights=False,
        )
        return self.dropout1(x2)

    def _ff_block(self, x):
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x2)
