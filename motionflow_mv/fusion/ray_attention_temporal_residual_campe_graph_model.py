"""Temporal ray-aware fusion with camera PE, residual refinement and graph joint relation."""

from typing import List, Tuple

import torch
import torch.nn as nn

from .ray_attention_temporal_residual_campe_model import RayAttentionFusionModelTemporalResidualCamPE
from .graph_joint_relation import GraphJointRelation, build_edge_index, H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS
from .ray_attention_model import _compute_rays


class RayAttentionFusionModelTemporalResidualCamPEGraph(RayAttentionFusionModelTemporalResidualCamPE):
    """Temporal ray-aware fusion with camera PE, residual refinement and graph joint relation.

    Replaces the dense ``joint_attn`` Transformer with a sparse
    :class:`GraphJointRelation` module that respects skeleton topology.

    Parameters
    ----------
    parents:
        Parent indices for the skeleton.  Default is the H36M 17-joint skeleton.
    symmetry_pairs:
        Left/right symmetry pairs.  Default is the H36M 17-joint pairs.
    graph_layers:
        Number of GJR message-passing layers.
    """

    def __init__(
        self,
        j: int = 17,
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
        # Bypass the parent's default joint_attn by passing n_joint_layers=0, then we add GJR ourselves.
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
            parents = H36M_17_PARENTS
        if symmetry_pairs is None:
            symmetry_pairs = H36M_17_SYMMETRY_PAIRS
        self.parents = parents
        self.symmetry_pairs = symmetry_pairs
        self.graph = GraphJointRelation(d=d, n_views=n_views, num_layers=graph_layers)
        self.register_buffer(
            "edge_index",
            build_edge_index(parents, symmetry_pairs, n_views, j)[0],
            persistent=False,
        )
        self.register_buffer(
            "edge_type",
            build_edge_index(parents, symmetry_pairs, n_views, j)[1],
            persistent=False,
        )

    def _extract_frame_features(self, x, K, R, t):
        """Run the per-frame encoder with camera PE and graph joint relation."""
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
        feat_v = feat_v.view(N, V, J, self.d)

        # Graph joint relation (replaces dense joint_attn).
        feat_j = self.graph(feat_v, self.edge_index, self.edge_type)

        return feat_j


if __name__ == "__main__":
    from .ray_attention_temporal_residual_model import _make_cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEGraph(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("CamPE + GraphJR temporal residual model sanity check passed")
