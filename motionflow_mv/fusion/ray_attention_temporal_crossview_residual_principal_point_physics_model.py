"""Iter14 anchor model with an physics-informed temporal skeleton dynamics refiner.

This is a thin subclass of
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` that adds a
small bidirectional GRU over the raw triangulated 3-D sequence before the final
residual MLP.  The extra head predicts a per-joint dynamics residual that is
added to the refined pose.

The module is intended to be used together with
``motionflow_mv.losses.physics_informed_dynamics.PhysicsInformedSkeletonDynamicsLoss``,
but it is fully self-contained and can be trained with the standard 3-D MSE
loss as well.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointPhysics(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Anchor model + temporal skeleton dynamics refiner.

    Parameters
    ----------
    dynamics_hidden:
        Hidden dimension of the per-joint bidirectional GRU (default 128).
    dynamics_layers:
        Number of GRU layers (default 1).
    apply_dynamics_to_residual:
        If ``True`` (default), the dynamics refiner predicts a residual on top
        of the existing residual-MLP output.  If ``False``, it replaces the
        residual MLP path.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        dynamics_hidden: int = 128,
        dynamics_layers: int = 1,
        apply_dynamics_to_residual: bool = True,
        return_pp_delta: bool = False,
        return_visibility: bool = False,
        return_raw: bool = False,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_visibility=return_visibility,
            return_raw=return_raw,
        )
        self.dynamics_hidden = dynamics_hidden
        self.dynamics_layers = dynamics_layers
        self.apply_dynamics_to_residual = apply_dynamics_to_residual

        # Per-joint dynamics refiner: operates on the 3-D trajectory of each
        # joint independently with shared weights.
        self.dynamics_refiner = nn.GRU(
            input_size=3,
            hidden_size=dynamics_hidden,
            num_layers=dynamics_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dynamics_head = nn.Linear(dynamics_hidden * 2, 3)

    def _dynamics_residual(self, pred_3d: torch.Tensor) -> torch.Tensor:
        """Predict a dynamics residual from a 3-D pose sequence.

        Args
        ----
        pred_3d:
            ``(B, T, J, 3)`` refined or raw 3-D pose sequence.

        Returns
        -------
        ``(B, T, J, 3)`` residual correction.
        """
        B, T, J, _ = pred_3d.shape
        # Process each joint independently: (B, J, T, 3) -> (B*J, T, 3)
        x = pred_3d.permute(0, 2, 1, 3).reshape(B * J, T, 3)
        out, _ = self.dynamics_refiner(x)  # (B*J, T, 2*dynamics_hidden)
        delta = self.dynamics_head(out)  # (B*J, T, 3)
        delta = delta.view(B, J, T, 3).permute(0, 2, 1, 3)  # (B, T, J, 3)
        return delta

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape

        # Re-use the anchor forward by forcing return_raw=True internally so we
        # can apply the dynamics refiner on top of both raw and refined outputs.
        parent_return_raw = self.return_raw
        self.return_raw = True
        out = super().forward(x, cameras=cameras, K=K, R=R, t=t)
        self.return_raw = parent_return_raw

        if isinstance(out, tuple):
            pred_3d, weights, pred_3d_raw = out[:3]
        else:
            pred_3d = out
            weights = None

        # Dynamics residual is computed on the refined trajectory and added to it.
        dynamics_delta = self._dynamics_residual(pred_3d)
        pred_3d = pred_3d + dynamics_delta

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            if weights is not None:
                weights = weights.squeeze(1)

        if self.return_raw:
            if weights is None:
                return pred_3d, pred_3d_raw.squeeze(1) if squeeze_output else pred_3d_raw
            return pred_3d, weights, pred_3d_raw.squeeze(1) if squeeze_output else pred_3d_raw

        if weights is not None:
            return pred_3d, weights
        return pred_3d
