from .triangulation import triangulate_dlt, triangulate_confidence_weighted
from .fusion_module import DLTFusion, FusionModule, FusionModuleRegistry, FUSION_REGISTRY
from .attention_fusion_module import AttentionFusionModule, register_attention_fusion_module
from .attention_fusion_v2_module import AttentionFusionV2Module, register_attention_v2_fusion_module
from .robust_triangulation_module import RobustTriangulationFusion, register_robust_triangulation_fusion_module
from .residual_refiner_module import ResidualRefinerFusion, register_residual_refiner_fusion_module
from .temporal_refiner_module import TemporalRefinerFusion, register_temporal_refiner_fusion_module
from .ray_attention_module import RayAttentionFusionModule, register_ray_attention_fusion_module

register_attention_fusion_module()
# register_attention_v2_fusion_module()  # experimental, training unstable
register_robust_triangulation_fusion_module()
register_residual_refiner_fusion_module()
register_temporal_refiner_fusion_module()
register_ray_attention_fusion_module()

__all__ = [
    "triangulate_dlt",
    "triangulate_confidence_weighted",
    "FusionModule",
    "DLTFusion",
    "AttentionFusionModule",
    "RobustTriangulationFusion",
    "ResidualRefinerFusion",
    "TemporalRefinerFusion",
    "FusionModuleRegistry",
    "FUSION_REGISTRY",
]
