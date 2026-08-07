"""Convenience re-export of the SSL cross-view contrastive fusion model.

See ``motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_ssl_view_contrast_model.py``
for the full implementation.  This module exists so that the variant is also
available under ``motionflow_mv.models``.
"""

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_ssl_view_contrast_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSSLViewContrast,
)

__all__ = ["RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSSLViewContrast"]
