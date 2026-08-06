"""PP-graph model with a skeleton-graph residual refiner.

Extends ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph``
by replacing its dense ``residual_mlp`` with a
``SkeletonGraphResidualRefiner`` that propagates pose corrections along the
bone and symmetry graph.
"""

from .ray_attention_temporal_crossview_residual_principal_point_graph_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph,
)
from .skeleton_graph_residual_refiner import SkeletonGraphResidualRefiner


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph,
):
    """Anchor PP-graph model with skeleton-graph residual refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the dense per-joint residual MLP with a skeleton-graph refiner.
        self.residual_mlp = SkeletonGraphResidualRefiner(
            j=self.j,
            in_dim=self.d + 3,
            hidden_dim=self.residual_hidden,
            num_layers=2,
        )
