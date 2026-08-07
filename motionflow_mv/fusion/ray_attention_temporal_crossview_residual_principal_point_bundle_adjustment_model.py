"""Anchor model with differentiable bundle-adjustment refinement.

Subclasses ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
and adds a lightweight, differentiable structure-only bundle-adjustment step
after the residual refinement head.  The DBA layer explicitly minimizes the
weighted reprojection error of the predicted 3D skeleton with respect to the
(corrected) cameras, improving physical alignment and robustness to small
calibration errors.
"""

from .differentiable_bundle_adjustment import DifferentiableBundleAdjustment
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBundleAdjustment(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Temporal + cross-view residual model with principal-point and DBA refinement.

    Parameters
    ----------
    dba_iters:
        Number of bundle-adjustment iterations (default 2).
    dba_damping:
        Levenberg-Marquardt damping (default 1.0).
    dba_max_update:
        Maximum 3D update in meters per BA iteration (default 0.05).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
    for the remaining arguments.
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
        return_raw: bool = False,
        dba_iters: int = 2,
        dba_damping: float = 1.0,
        dba_max_update: float = 0.05,
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
        self.dba_iters = dba_iters
        self.dba = DifferentiableBundleAdjustment(
            n_iters=dba_iters,
            damping=dba_damping,
            max_update=dba_max_update,
        )

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        # Run the anchor forward on the raw input to get its refined 3D output and
        # per-view weights, then post-process with a differentiable bundle
        # adjustment step.
        squeeze_output = x.dim() == 4
        x_sequence = x.unsqueeze(1) if squeeze_output else x

        B, T, V, J, _ = x_sequence.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors

            K, R, t = _cameras_to_tensors(cameras, device)

        # Keep original camera tensors for later DBA; the anchor expects either
        # (V, ...) or (B, V, ...) and broadcasts internally.
        K_orig, R_orig, t_orig = K, R, t

        # Run anchor forward.  It handles all camera broadcasting internally and
        # returns (pred_3d, weights, [optional extras]).
        pred_3d, weights, *extras = super().forward(x, K=K, R=R, t=t)

        # Broadcast cameras the same way the anchor does, so shapes match (B, T, V, ...).
        if K_orig.dim() == 3:
            K_bt = K_orig.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1, -1)
            R_bt = R_orig.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1, -1)
            t_bt = t_orig.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1)
        else:
            K_bt = K_orig.unsqueeze(1).expand(B, T, -1, -1, -1)
            R_bt = R_orig.unsqueeze(1).expand(B, T, -1, -1, -1)
            t_bt = t_orig.unsqueeze(1).expand(B, T, -1, -1)

        x_flat = x_sequence.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2].view(B, T, V, J, 2)

        pred_3d_bt = pred_3d.view(B, T, J, 3)
        weights_bt = weights.view(B, T, V, J)

        pred_3d_refined = self.dba(
            pred_3d_bt,
            points_2d,
            K_bt,
            R_bt,
            t_bt,
            weights_bt,
        )

        if squeeze_output:
            pred_3d_refined = pred_3d_refined.squeeze(1)

        return (pred_3d_refined, weights, *extras)
