"""Mixed-dataset temporal ray-attention fusion with residual refinement.

Extends ``RayAttentionFusionModelTemporalMixed`` (mixed_v1) by adding a
lightweight dataset-specific residual refinement head on top of the raw DLT
triangulated 3D output.  The shared backbone still handles MPI-INF-3DHP,
AIST++, and Human3.6M with different view/joint counts by padding to a common
grid and using per-dataset output heads.

Supported datasets
------------------
* ``mpi``   : 14 views, 28 joints (MPI-INF-3DHP)
* ``aist``  :  9 views, 17 joints (AIST++)
* ``h36m``  :  4 views, 17 joints (Human3.6M)
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_model_mixed_v1 import RayAttentionFusionModelTemporalMixed
from .ray_attention_model import _triangulate_weighted_dlt


class RayAttentionFusionModelTemporalMixedResidual(RayAttentionFusionModelTemporalMixed):
    """Mixed-dataset temporal fusion with residual refinement.

    Parameters
    ----------
    d:
        Feature dimension of the shared backbone (default 64).
    n_temporal_layers:
        Number of temporal transformer layers (default 2).
    max_temporal_len:
        Maximum temporal sequence length for positional embeddings
        (default 256).
    residual_hidden:
        Hidden dimension of the residual MLP (default 128).
    """

    _DATASET_SPECS = {
        0: {"name": "mpi", "n_views": 14, "n_joints": 28},
        1: {"name": "aist", "n_views": 9, "n_joints": 17},
        2: {"name": "h36m", "n_views": 4, "n_joints": 17},
    }

    def __init__(
        self,
        d: int = 64,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
    ):
        # Store before super().__init__ because the parent reads _DATASET_SPECS.
        self.residual_hidden = residual_hidden
        super().__init__(
            d=d,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
        )

        # Per-dataset residual refinement heads.
        self.residual_mlps = nn.ModuleDict()
        for did, spec in self._DATASET_SPECS.items():
            self.residual_mlps[spec["name"]] = nn.Sequential(
                nn.Linear(d + 3, residual_hidden),
                nn.ReLU(),
                nn.Linear(residual_hidden, residual_hidden),
                nn.ReLU(),
                nn.Linear(residual_hidden, 3),
            )

    def _run_branch(self, feat, points_2d, confidences, P, name):
        """Run one dataset-specific head, triangulate, refine residual, and pad."""
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

        # Raw DLT triangulation.
        pred = _triangulate_weighted_dlt(points_2d_vj, weights, P_vj)  # (N*T, j, 3)

        # Residual refinement from pooled temporal features + raw 3D estimate.
        feat_pooled = feat_vj.mean(dim=1)  # (N*T, j, d)
        residual_input = torch.cat([feat_pooled, pred], dim=-1)  # (N*T, j, d+3)
        delta = self.residual_mlps[name](residual_input)  # (N*T, j, 3)
        pred = pred + delta

        # Pad to common joint dimension for loss masking.
        N = pred.shape[0]
        pred_pad = torch.zeros(N, self.max_joints, 3, device=pred.device, dtype=pred.dtype)
        pred_pad[:, :j, :] = pred
        mask = torch.zeros(N, self.max_joints, device=pred.device, dtype=torch.bool)
        mask[:, :j] = True
        return pred_pad, mask

    def forward(self, x, K, R, t, dataset_ids):
        """Multi-dataset forward that dispatches by dataset id dynamically."""
        if x.dim() == 4:
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

        for did, spec in self._DATASET_SPECS.items():
            idx = (dataset_ids == did).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            # Expand index across the T time steps.
            pos = (idx[:, None] * T + torch.arange(T, device=device)[None, :]).view(-1)
            pred_branch, mask_branch = self._run_branch(
                feat[pos], points_2d[pos], confidences[pos], P[pos], spec["name"]
            )
            pred_all[pos] = pred_branch
            mask_all[pos] = mask_branch

        pred_all = pred_all.view(B, T, self.max_joints, 3)
        mask_all = mask_all.view(B, T, self.max_joints)
        return pred_all, mask_all
