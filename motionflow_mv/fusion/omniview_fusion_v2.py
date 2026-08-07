"""OmniMultiViewFusion v2 — integrated multi-view fusion prototype.

Combines four previously isolated ideas into a single model:

1. **Visibility gating** – explicit per-view/per-joint occlusion mask with a
   fallback guard for degenerate views.
2. **Graph-joint attention** – anatomically constrained message passing over a
   sparse (view, joint) skeleton graph.
3. **Uncertainty-weighted triangulation** – learned anisotropic image-space
   covariance and adaptive Gauss-Newton refinement (Bayesian tri v2).
4. **Spatiotemporal transformer** – joint attention over time and views.

This file is intentionally self-contained and lives in ``motionflow_mv/fusion/``
as an isolated prototype. It subclasses the Bayesian tri v2 model and replaces
the dense joint attention with graph-joint attention, while adding an explicit
visibility head.

Input:
    x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output (tuple):
    pred_3d:   (B, T, J, 3) or (B, J, 3) world-coordinate 3D joints
    weights:   (B, T, V, J) or (B, V, J) per-view per-joint fusion weights
    visibility:(B, T, V, J) or (B, V, J) predicted visibility probabilities
    covariance:(B, T, V, J, 2, 2) or (B, V, J, 2, 2) predicted 2D covariances
    epipolar_loss: scalar auxiliary loss
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from motionflow_mv.fusion.graph_joint_attention_v2 import GraphJointAttentionV2
from motionflow_mv.fusion.graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq


class OmniMultiViewFusionV2(RayAttentionFusionModelBayesianTriV2):
    """OmniMultiViewFusion v2 prototype.

    Extends ``RayAttentionFusionModelBayesianTriV2`` with:

    * An explicit per-view/per-joint visibility head and fallback guard.
    * A graph-joint attention block in place of the dense joint attention
      layers in the per-frame encoder.

    Parameters
    ----------
    graph_num_layers:
        Number of cross-view graph attention layers (default 1).
    visibility_threshold:
        Probability below which a view is treated as occluded.
    min_visible_views:
        Minimum visible views before the fallback guard forces visibility=1.
    graph_dropout:
        Dropout on graph attention weights.
    See ``RayAttentionFusionModelBayesianTriV2`` for remaining args.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 0,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_covariance: bool = True,
        covariance_hidden: int = 64,
        gn_iters: int = 2,
        min_gn_damping: float = 1e-6,
        max_gn_damping: float = 1e-2,
        epipolar_loss_weight: float = 0.05,
        graph_num_layers: int = 1,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        graph_dropout: float = 0.0,
    ):
        # The dense joint-attention layers are handled here, not by the ancestor,
        # so we always ask the ancestor to create none of its own.
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=0,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_covariance=return_covariance,
            covariance_hidden=covariance_hidden,
            gn_iters=gn_iters,
            min_gn_damping=min_gn_damping,
            max_gn_damping=max_gn_damping,
            epipolar_loss_weight=epipolar_loss_weight,
        )

        # The Bayesian-tri ancestor forces return_pp_delta=True. Re-enable user
        # control so the output arity stays predictable.
        self.return_pp_delta = return_pp_delta

        self.visibility_threshold = visibility_threshold
        self.min_visible_views = max(2, min_visible_views)

        # Explicit visibility head: per (view, joint) soft visibility mask.
        self.visibility_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

        # Optional dense joint-level self-attention (per-view) for capacity.
        # Use a distinct attribute so we do not overwrite the ancestor's
        # (empty) joint_attn ModuleList.
        self.n_joint_layers = n_joint_layers
        if n_joint_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=d * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.omni_joint_attn = nn.ModuleList([layer for _ in range(n_joint_layers)])
        else:
            self.omni_joint_attn = None

        # Optional graph-joint attention block. When graph_num_layers == 0 the
        # module is omitted entirely so the no-graph ablation is truly graph-free.
        self.graph_num_layers = graph_num_layers
        if graph_num_layers > 0:
            self.graph_joint_attention = GraphJointAttentionV2(
                d=d,
                n_views=n_views,
                n_layers=graph_num_layers,
                n_heads=n_heads,
                n_edge_types=4,
                dropout=graph_dropout,
            )
            # Register a default skeleton graph for H36M 17 joints. For MPI-INF-3DHP
            # the graph can be rebuilt via ``rebuild_graph``.
            self._j = j
            self._current_j = j
            self.graph_joint_attention.build_edge_index(
                j=j,
                parents=H36M_17_PARENTS,
                symmetry_pairs=H36M_17_SYMMETRY_PAIRS,
                add_self_loops=True,
            )
        else:
            self.graph_joint_attention = None

    def rebuild_graph(self, j: int, dataset: str = "h36m") -> None:
        """Rebuild the (view, joint) graph for a different skeleton.

        Args:
            j: number of joints.
            dataset: one of "h36m" or "mpiinf3dhp".
        """
        self._current_j = j
        if dataset in ("h36m", "human3.6m", "h36m_17"):
            parents = H36M_17_PARENTS
            symmetry = H36M_17_SYMMETRY_PAIRS
        elif dataset in ("mpiinf3dhp", "mpi", "mpiinf3dhp_28"):
            parents = MPI_INF_3DHP_28_PARENTS
            symmetry = MPI_INF_3DHP_28_SYMMETRY_PAIRS
        else:
            raise ValueError(f"Unknown dataset for graph building: {dataset}")

        if self.graph_joint_attention is None:
            return
        self.graph_joint_attention.build_edge_index(
            j=j,
            parents=parents,
            symmetry_pairs=symmetry,
            add_self_loops=True,
        )

    def _apply_graph_joint_attention(
        self,
        feat: torch.Tensor,
        j: int,
    ) -> torch.Tensor:
        """Run graph-joint attention on per-frame features.

        Args:
            feat: (N, V, J, d)
            j: number of joints.

        Returns:
            (N, V, J, d)
        """
        if self.graph_joint_attention is None:
            return feat
        if j != self._current_j:
            # Auto-rebuild graph if the input joint count changes. This is a
            # convenience for variable-view / cross-dataset inference.
            self.rebuild_graph(j, dataset="h36m" if j == 17 else "mpiinf3dhp")

        return self.graph_joint_attention(feat)

    def _visibility_multiplier(
        self,
        feat: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-view/per-joint visibility multipliers in [0, 1].

        Args:
            feat: (N, V, J, d)
            confidences: (N, V, J)

        Returns:
            visibility: (N, V, J)
        """
        # Raw logits -> soft visibility.
        v_logits = self.visibility_head(feat).squeeze(-1)  # (N, V, J)
        visibility = torch.sigmoid(v_logits)

        # Fallback guard: if too few views are visible, force all views on.
        visible = (visibility > self.visibility_threshold).float()
        visible_count = visible.sum(dim=1)  # (N, J)
        fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)  # (N, 1, J)
        effective_visibility = visibility + (1.0 - visibility) * fallback

        # Still suppress completely zero-confidence observations.
        return effective_visibility * (confidences > 1e-6).float()

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[object] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, ...]:
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _cameras_to_tensors
            K, R, t = _cameras_to_tensors(cameras, device)

        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # ---- NEW: dense joint-level self-attention (optional capacity) ----
        if self.omni_joint_attn is not None:
            feat_j = feat.permute(0, 2, 1, 3).reshape(B * T * V, J, self.d)
            for layer in self.omni_joint_attn:
                feat_j = layer(feat_j)
            feat = feat_j.view(B * T, V, J, self.d)

        # ---- NEW: graph-joint attention over (view, joint) skeleton graph ----
        feat = self._apply_graph_joint_attention(feat, J)  # (B*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Anisotropic covariance prediction per (view, joint).
        raw_cov = self.covariance_head(feat)  # (B*T, V, J, 3)
        L = self._cholesky_to_covariance(raw_cov)  # (B*T, V, J, 2, 2)

        # Precision weight = 1 / sqrt(det(Σ)) = 1 / (l_xx * l_yy).
        precision = 1.0 / (
            L[..., 0, 0].clamp(min=1e-4) * L[..., 1, 1].clamp(min=1e-4)
        )

        # ---- NEW: explicit visibility gating ----
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * precision * visibility  # (B*T, V, J)
        # Clamp to avoid extreme weights that can destabilise DLT/GN.
        weights = weights.clamp(min=1e-4, max=1e4)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt

        # Fully batched DLT.
        pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights)

        # Adaptive Gauss-Newton refinement.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        damping = self.damping_head(feat_pooled).squeeze(-1)  # (B*T, J)
        damping = self.min_gn_damping + (
            self.max_gn_damping - self.min_gn_damping
        ) * damping

        from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
            _adaptive_gauss_newton,
        )

        pred_3d_gn = _adaptive_gauss_newton(
            points_2d,
            weights,
            K_corrected,
            R,
            t,
            pred_3d_raw,
            damping,
            num_iters=self.gn_iters,
        )

        # Residual refinement head.
        residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_gn + delta

        # Epipolar consistency loss.
        epi_loss = self._epipolar_consistency_loss(points_2d, K_corrected, R, t, L)
        epi_loss = self.epipolar_loss_weight * epi_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        L = L.view(B, T, V, J, 2, 2)
        visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            L = L.squeeze(1)
            visibility = visibility.squeeze(1)

        # Build output tuple.
        out = (pred_3d, weights, visibility, L, epi_loss)

        if self.return_pp_delta:
            out += (pp_delta,)
            if self.correct_focal:
                out += (focal_scale,)

        return out


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for smoke tests)."""
    from motionflow_mv.calibration.camera import Camera

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
    # CPU smoke test: shape/gradient sanity + single-frame compatibility.
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = OmniMultiViewFusionV2(j=J, d=64, n_views=V, graph_num_layers=1)
    pred, weights, visibility, covariance, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)
    assert covariance.shape == (B, T, V, J, 2, 2)
    assert epi_loss.numel() == 1

    loss = pred.mean() + epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("OmniMultiViewFusionV2 multi-frame sanity check passed")

    # Single-frame compatibility.
    x4 = torch.rand(B, V, J, 3)
    pred4, weights4, visibility4, covariance4, epi4 = model(x4, cameras=cameras)
    assert pred4.shape == (B, J, 3)
    assert weights4.shape == (B, V, J)
    assert visibility4.shape == (B, V, J)
    assert covariance4.shape == (B, V, J, 2, 2)
    print("OmniMultiViewFusionV2 single-frame sanity check passed")
