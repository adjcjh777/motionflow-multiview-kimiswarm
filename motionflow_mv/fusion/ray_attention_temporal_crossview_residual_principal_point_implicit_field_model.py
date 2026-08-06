"""PP-corrected anchor with a neural implicit 3-D pose-field residual refiner.

Subclasses ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
and replaces its dense per-joint residual MLP with a
``NeuralImplicitPoseFieldRefiner``.  The refiner treats the raw triangulated
3-D pose as an initial estimate and walks it toward the zero level-set of a
joint-conditioned neural implicit field.
"""

from .neural_implicit_pose_field import NeuralImplicitPoseFieldRefiner
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointImplicitField(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Anchor PP model with an implicit-field residual refiner.

    Parameters
    ----------
    field_hidden:
        Hidden width of the implicit field MLP (default 128).
    field_layers:
        Number of layers in the implicit field MLP (default 3).
    field_iters:
        Number of Newton-style field-refinement steps (default 1).
    field_step_size:
        Step-size multiplier for each Newton step (default 0.5).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
    """

    def __init__(
        self,
        *args,
        field_hidden: int = 128,
        field_layers: int = 3,
        field_iters: int = 1,
        field_step_size: float = 0.5,
        return_field: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_field = return_field
        self.residual_mlp = NeuralImplicitPoseFieldRefiner(
            j=self.j,
            feat_dim=self.d,
            hidden_dim=field_hidden,
            num_layers=field_layers,
            n_iters=field_iters,
            step_size=field_step_size,
            return_field=return_field,
        )
