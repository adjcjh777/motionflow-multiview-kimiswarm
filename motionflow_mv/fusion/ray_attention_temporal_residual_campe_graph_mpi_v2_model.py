"""Temporal ray-aware fusion with camera PE, residual refinement and MPI-INF-3DHP graph.

This model is a specialization of
``RayAttentionFusionModelTemporalResidualCamPEGraph`` for the full 28-joint
MPI-INF-3DHP skeleton.  It hard-codes the 28-joint parent/symmetry graph,
adds lightweight skeleton-aware regularization helpers, and uses an improved
edge-conditioned graph message-passing block that is vectorised over edge
types.

The intended use is the MPI-INF-3DHP benchmark (``j=28``), but the model
remains generic enough for any 28-joint multi-view pose problem whose
skeleton matches the MPI layout.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .ray_attention_temporal_residual_campe_model import RayAttentionFusionModelTemporalResidualCamPE
from .graph_joint_relation import (
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    build_edge_index,
)
from ..calibration.camera import Camera


class GraphJointRelationMPIV2(nn.Module):
    """Improved edge-conditioned graph message passing for the MPI 28-joint skeleton.

    Compared with ``GraphJointRelation``:

    * Edge projections are applied in a single vectorised batched linear call
      rather than looping over edge types.
    * Edge-type embeddings is added to the attention gate so bone/symmetry/
      cross-view edges can use different gating dynamics.
    * A residual connection and layer norm are applied per message-passing step.

    Input shape:  (B, V, J, d)
    Output shape: (B, V, J, d)
    """

    def __init__(self, d: int = 64, n_views: int = 4, num_layers: int = 3):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        self.n_views = n_views

        # Three edge types: bone, symmetry, cross-view.
        self.edge_proj = nn.Linear(d, d)
        self.edge_type_embed = nn.Embedding(3, d)

        self.edge_attn = nn.Sequential(
            nn.Linear(d * 2 + d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        B, V, J, _ = x.shape
        h = x.view(B * V * J, self.d)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        type_emb = self.edge_type_embed(edge_type)  # (E, d)

        for layer_idx in range(self.num_layers):
            src = h[src_idx]
            dst = h[dst_idx]

            # Edge attention conditioned on src, dst and edge type embedding.
            attn_input = torch.cat([src, dst, type_emb], dim=-1)
            attn = torch.sigmoid(self.edge_attn(attn_input)).squeeze(-1)  # (E,)

            # Project source features and gate by attention.
            projected = self.edge_proj(src)
            msg = attn.unsqueeze(-1) * projected

            # Aggregate messages to destination nodes.
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst_idx, msg)

            h = self.norms[layer_idx](h + agg)

        return h.view(B, V, J, self.d)


class RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2(RayAttentionFusionModelTemporalResidualCamPE):
    """CamPE + residual + graph model hard-coded for the MPI-INF-3DHP 28-joint skeleton.

    Parameters
    ----------
    parents:
        Parent indices for the skeleton.  Defaults to ``MPI_INF_3DHP_28_PARENTS``.
    symmetry_pairs:
        Left/right symmetry pairs.  Defaults to
        ``MPI_INF_3DHP_28_SYMMETRY_PAIRS``.
    graph_layers:
        Number of GJR message-passing layers.
    """

    def __init__(
        self,
        j: int = 28,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        n_bands: int = 4,
        parents: List[int] = None,
        symmetry_pairs: List[Tuple[int, int]] = None,
        graph_layers: int = 3,
        **kwargs,
    ):
        # Bypass the dense joint attention from the CamPE base.
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=0,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            n_bands=n_bands,
        )

        if parents is None:
            parents = MPI_INF_3DHP_28_PARENTS
        if symmetry_pairs is None:
            symmetry_pairs = MPI_INF_3DHP_28_SYMMETRY_PAIRS

        self.parents = parents
        self.symmetry_pairs = symmetry_pairs
        self.graph_layers = graph_layers

        # Sanity check: provided skeleton must match joint count.
        if len(self.parents) != j:
            raise ValueError(
                f"Skeleton parent list has {len(self.parents)} entries, expected j={j}."
            )

        self.graph = GraphJointRelationMPIV2(d=d, n_views=n_views, num_layers=graph_layers)
        edge_index, edge_type = build_edge_index(
            self.parents, self.symmetry_pairs, n_views=n_views, j=j, add_self_loops=True
        )
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_type", edge_type, persistent=False)

    def _bone_vectors(self, X: torch.Tensor) -> torch.Tensor:
        """Return parent->child bone vectors for the current 3D pose.

        Args:
            X: (..., J, 3) 3D joints. Any leading batch/time dimensions are
            preserved.

        Returns:
            (..., J, 3) bone vectors; entries for root joints are zero.
        """
        J = X.shape[-2]
        bones = torch.zeros_like(X)
        for child, parent in enumerate(self.parents):
            if parent >= 0:
                bones[..., child, :] = X[..., child, :] - X[..., parent, :]
        return bones

    def bone_length_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 bone-length error between prediction and target poses.

        Args:
            pred: (..., J, 3) predicted 3D joints.
            target: (..., J, 3) ground-truth 3D joints.

        Returns:
            Scalar mean bone-length error.
        """
        pred_bones = self._bone_vectors(pred)
        target_bones = self._bone_vectors(target)
        return (pred_bones.norm(dim=-1) - target_bones.norm(dim=-1).detach()).abs().mean()

    def symmetry_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalty on length differences between mirrored limb pairs.

        Args:
            pred: (..., J, 3) predicted 3D joints.

        Returns:
            Scalar mean symmetry error.
        """
        bones = self._bone_vectors(pred)
        bone_lengths = bones.norm(dim=-1)  # (..., J)
        errors = []
        for left, right in self.symmetry_pairs:
            errors.append((bone_lengths[..., left] - bone_lengths[..., right]).abs())
        if not errors:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.stack(errors, dim=-1).mean()

    def _extract_frame_features(self, x, K, R, t):
        """Run the per-frame encoder with camera PE and MPI graph joint relation."""
        from .ray_attention_model import _compute_rays

        N, V, J, _ = x.shape
        points_2d = x[..., :2]

        rays = _compute_rays(points_2d, K, R, t)
        obs_emb = self.obs_embed(x)
        ray_emb = self.ray_embed(torch.cat([torch.zeros_like(rays), rays], dim=-1))
        feat = torch.cat([obs_emb, ray_emb], dim=-1)

        # Camera positional encoding.
        camera_emb = self.camera_embed_mlp(K, R, t)
        camera_emb = camera_emb[:, :, None, :].expand(N, V, J, self.d)
        feat = feat + camera_emb

        # View-level attention.
        feat_v = feat.permute(0, 2, 1, 3).reshape(N * J, V, self.d)
        attn_out, _ = self.view_attn(feat_v, feat_v, feat_v)
        feat_v = self.view_norm1(feat_v + attn_out)
        feat_v = self.view_norm2(feat_v + self.view_ffn(feat_v))
        feat_v = feat_v.view(N, J, V, self.d)

        # Graph joint relation (replaces dense joint_attn).
        feat_j = self.graph(feat_v, self.edge_index, self.edge_type)

        return feat_j


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for tests)."""
    import numpy as np
    from ..calibration.camera import Camera

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


if __name__ == "__main__":
    B, T, V, J = 2, 5, 4, 28
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2(
        j=J, d=64, n_views=V
    )
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("CamPE + MPI Graph v2 model sanity check passed")

    # Skeleton-aware loss sanity check.
    y = torch.rand_like(pred)
    bl_loss = model.bone_length_loss(pred, y)
    sym_loss = model.symmetry_loss(pred)
    assert bl_loss.item() >= 0
    assert sym_loss.item() >= 0
    print(f"bone_length_loss={bl_loss.item():.6f}, symmetry_loss={sym_loss.item():.6f}")
