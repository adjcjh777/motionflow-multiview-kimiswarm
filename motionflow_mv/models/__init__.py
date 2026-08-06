"""High-level model wrappers and utilities."""

from .crossview_residual_uncertainty import CrossviewResidualUncertaintyModel
from .domain_adaptation_wrapper import DomainAdaptationWrapper

try:
    from .shelf_campus_domain_adaptation import ShelfCampusDomainAdaptationWrapper
except ImportError:  # pragma: no cover
    ShelfCampusDomainAdaptationWrapper = None  # type: ignore


try:
    from .spatiotemporal_principal_point_model import SpatiotemporalPrincipalPointModel
except ImportError:  # pragma: no cover
    SpatiotemporalPrincipalPointModel = None  # type: ignore

try:
    from .crossview_residual_visibility_v2 import CrossviewResidualVisibilityV2
except ImportError:  # pragma: no cover
    CrossviewResidualVisibilityV2 = None  # type: ignore

try:
    from .spatial_feature_pyramid import SpatialFeaturePyramid, SpatialFeaturePyramidModel
except ImportError:  # pragma: no cover
    SpatialFeaturePyramid = None  # type: ignore
    SpatialFeaturePyramidModel = None  # type: ignore

try:
    from .graph_joint_relation import (
        GraphJointRelation,
        build_edge_index,
        H36M_17_PARENTS,
        H36M_17_SYMMETRY_PAIRS,
        MPI_INF_3DHP_28_PARENTS,
        MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    )
except ImportError:  # pragma: no cover
    GraphJointRelation = None  # type: ignore
    build_edge_index = None  # type: ignore
    H36M_17_PARENTS = None  # type: ignore
    H36M_17_SYMMETRY_PAIRS = None  # type: ignore
    MPI_INF_3DHP_28_PARENTS = None  # type: ignore
    MPI_INF_3DHP_28_SYMMETRY_PAIRS = None  # type: ignore

try:
    from .distilled_student_principal_point_model import DistilledStudentPrincipalPointModel
except ImportError:  # pragma: no cover
    DistilledStudentPrincipalPointModel = None  # type: ignore

try:
    from .data_augmentation_multiview_wrapper import MultiViewDataAugmentationWrapper
except ImportError:  # pragma: no cover
    MultiViewDataAugmentationWrapper = None  # type: ignore

__all__ = [
    "CrossviewResidualUncertaintyModel",
    "CrossviewResidualVisibilityV2",
    "DistilledStudentPrincipalPointModel",
    "DomainAdaptationWrapper",
    "ShelfCampusDomainAdaptationWrapper",
    "GraphJointRelation",
    "build_edge_index",
    "H36M_17_PARENTS",
    "H36M_17_SYMMETRY_PAIRS",
    "MPI_INF_3DHP_28_PARENTS",
    "MPI_INF_3DHP_28_SYMMETRY_PAIRS",
    "MultiViewDataAugmentationWrapper",
]
