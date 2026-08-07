"""Mixed-dataset temporal ray-attention fusion model.

This wrapper trains a single shared temporal backbone on datasets that may have
different numbers of views and/or joints.  It pads each clip to a common
``(max_views, max_joints)`` grid, runs the per-frame + temporal encoder from
``RayAttentionFusionModelTemporal``, then branches to dataset-specific view/
joint heads for triangulation.

The current implementation supports two dataset branches:

* ``mpi``     : 14 views, 28 joints (MPI-INF-3DHP)
* ``h36m``    : 4 views, 17 joints (Human3.6M)

No existing code is modified; the original ``RayAttentionFusionModelTemporal``
is reused only as a feature extractor.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_model import RayAttentionFusionModelTemporal
from .ray_attention_model import _triangulate_weighted_dlt


class RayAttentionFusionModelTemporalMixed(nn.Module):
    """Shared temporal backbone with per-dataset output heads.

    Input:
        x: (B, T, Vmax, Jmax, 3)  -- (pixel_x, pixel_y, confidence)
        K, R, t: (B, Vmax, 3, 3), (B, Vmax, 3, 3), (B, Vmax, 3)
        dataset_ids: (B,) int -- 0 for MPI, 1 for H36M

    Output:
        pred_3d: (B, T, Jmax, 3)
        joint_mask: (B, T, Jmax) bool -- which joints are valid per sample
    """

    _DATASET_SPECS = {
        0: {"name": "mpi", "n_views": 14, "n_joints": 28},
        1: {"name": "h36m", "n_views": 4, "n_joints": 17},
    }

    def __init__(self, d: int = 64, n_temporal_layers: int = 2, max_temporal_len: int = 256):
        super().__init__()
        self.d = d
        self.max_views = max(spec["n_views"] for spec in self._DATASET_SPECS.values())
        self.max_joints = max(spec["n_joints"] for spec in self._DATASET_SPECS.values())

        # Shared backbone: per-frame v3 encoder + temporal transformer.
        self.backbone = RayAttentionFusionModelTemporal(
            j=self.max_joints,
            d=d,
            n_views=self.max_views,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
        )

        # Per-dataset fusion MLPs and weight heads.
        self.fusion_mlps = nn.ModuleDict()
        self.weight_heads = nn.ModuleDict()
        for did, spec in self._DATASET_SPECS.items():
            v = spec["n_views"]
            self.fusion_mlps[spec["name"]] = nn.Sequential(
                nn.Linear(d * v, d),
                nn.ReLU(),
                nn.Linear(d, d),
            )
            self.weight_heads[spec["name"]] = nn.Linear(d, 1)

    def _temporal_features(self, x, K, R, t):
        """Run shared per-frame + temporal encoder."""
        B, T, V, J, _ = x.shape
        assert (V, J) == (self.max_views, self.max_joints), "Input must be padded to max dims"

        # Per-frame features.  The underlying helper expects (N, V, J, 3) with
        # per-sample camera tensors of shape (N, V, ...).
        x_flat = x.reshape(B * T, V, J, 3)
        K = K.unsqueeze(1).expand(B, T, V, 3, 3).reshape(B * T, V, 3, 3)
        R = R.unsqueeze(1).expand(B, T, V, 3, 3).reshape(B * T, V, 3, 3)
        t = t.unsqueeze(1).expand(B, T, V, 3).reshape(B * T, V, 3)
        feat = self.backbone._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Temporal transformer.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.backbone.temporal_pos_embed[:T]
        for layer in self.backbone.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)
        return feat

    def _run_branch(self, feat, points_2d, confidences, P, name):
        """Run one dataset-specific head and triangulate.

        All tensors are sub-batches already sliced by dataset index.
        """
        spec = next(s for s in self._DATASET_SPECS.values() if s["name"] == name)
        v = spec["n_views"]
        j = spec["n_joints"]

        feat_vj = feat[:, :v, :j, :]
        points_2d_vj = points_2d[:, :v, :j, :]
        conf_vj = confidences[:, :v, :j]
        P_vj = P[:, :v, :, :]

        # Predict per-view weights.
        feat_perm = feat_vj.permute(0, 2, 1, 3)  # (N*T, j, v, d)
        w_logits = self.weight_heads[name](feat_perm).squeeze(-1)  # (N*T, j, v)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (N*T, v, j)
        weights = weights * conf_vj

        pred = _triangulate_weighted_dlt(points_2d_vj, weights, P_vj)  # (N*T, j, 3)

        # Pad to common joint dimension.
        N = pred.shape[0]
        pred_pad = torch.zeros(N, self.max_joints, 3, device=pred.device, dtype=pred.dtype)
        pred_pad[:, :j, :] = pred
        mask = torch.zeros(N, self.max_joints, device=pred.device, dtype=torch.bool)
        mask[:, :j] = True
        return pred_pad, mask

    def forward(self, x, K, R, t, dataset_ids):
        squeeze_output = x.dim() == 4
        if squeeze_output:
            x = x.unsqueeze(1)
        B, T, V, J, _ = x.shape
        assert (V, J) == (self.max_views, self.max_joints)
        device = x.device

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Shared temporal features.
        feat = self._temporal_features(x, K, R, t)  # (B*T, V, J, d)

        # Projection matrices for all views (same rig across the clip).
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B, V, 3, 4)
        P = K @ Rt  # (B, V, 3, 4)
        P = P.unsqueeze(1).expand(B, T, V, 3, 4).reshape(B * T, V, 3, 4)

        pred_all = torch.zeros(B * T, self.max_joints, 3, device=device, dtype=x.dtype)
        mask_all = torch.zeros(B * T, self.max_joints, device=device, dtype=torch.bool)

        dataset_ids = dataset_ids.to(device)

        # MPI branch.
        mpi_idx = (dataset_ids == 0).nonzero(as_tuple=True)[0]
        if len(mpi_idx):
            # Expand index across the T time steps.
            pos = (mpi_idx[:, None] * T + torch.arange(T, device=device)[None, :]).view(-1)
            pred_mpi, mask_mpi = self._run_branch(
                feat[pos], points_2d[pos], confidences[pos], P[pos], "mpi"
            )
            pred_all[pos] = pred_mpi
            mask_all[pos] = mask_mpi

        # H36M branch.
        h36m_idx = (dataset_ids == 1).nonzero(as_tuple=True)[0]
        if len(h36m_idx):
            pos = (h36m_idx[:, None] * T + torch.arange(T, device=device)[None, :]).view(-1)
            pred_h36m, mask_h36m = self._run_branch(
                feat[pos], points_2d[pos], confidences[pos], P[pos], "h36m"
            )
            pred_all[pos] = pred_h36m
            mask_all[pos] = mask_h36m

        pred_all = pred_all.view(B, T, self.max_joints, 3)
        mask_all = mask_all.view(B, T, self.max_joints)
        if squeeze_output:
            pred_all = pred_all.squeeze(1)
            mask_all = mask_all.squeeze(1)
        return pred_all, mask_all
