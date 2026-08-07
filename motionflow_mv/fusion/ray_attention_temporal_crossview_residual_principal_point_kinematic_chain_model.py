"""Anchor PP model with a kinematic-chain graph convolutional refiner.

This subclasses the current best model
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` and
appends a ``KinematicChainGraphRefiner`` to the output 3-D skeleton.  It is a
minimal, reversible change: every other component (ray embedding, spatio-temporal
transformer, weight head, principal-point correction, residual MLP) is inherited
unchanged.
"""

import torch.nn as nn

from .kinematic_chain_graph_refiner import KinematicChainGraphRefinerTemporal
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual + PP + kinematic-chain graph refiner.

    Parameters
    ----------
    kc_hidden_dim:
        Hidden dimension of the kinematic-chain graph refiner (default 64).
    kc_num_layers:
        Number of graph message-passing layers in the refiner (default 2).
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
        return_pp_delta: bool = False,
        return_visibility: bool = False,
        kc_hidden_dim: int = 64,
        kc_num_layers: int = 2,
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
        )
        self.focal_max_scale = focal_max_scale
        self.kc_hidden_dim = kc_hidden_dim
        self.kc_num_layers = kc_num_layers
        self.kinematic_chain_refiner = KinematicChainGraphRefinerTemporal(
            j=j,
            hidden_dim=kc_hidden_dim,
            num_layers=kc_num_layers,
            share_weights=True,
        )

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        out = super().forward(x, cameras=cameras, K=K, R=R, t=t)

        # Apply the kinematic-chain graph refiner on the output skeleton.
        pred_3d_refined = self.kinematic_chain_refiner(out[0])
        return (pred_3d_refined, *out[1:])
