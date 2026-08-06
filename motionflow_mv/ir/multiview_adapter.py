"""Adapter that fuses per-view ``HumanMotionIR`` instances into a single IR.

The adapter is intentionally thin: it canonicalizes per-view observations,
calls a ``FusionModule`` to obtain world-coordinate 3D joints, and repackages
the result into a new ``HumanMotionIR`` that downstream consumers can treat
like any other single-view IR.
"""

from typing import List

import numpy as np

from ..calibration.camera import Camera
from ..fusion.fusion_module import FusionModule
from .human_motion_ir import HumanMotionIR


def fuse_multiple_irs(
    irs: List[HumanMotionIR],
    cameras: List[Camera],
    fusion_module: FusionModule,
    fused_sequence_id: str = "",
) -> HumanMotionIR:
    """Fuse a list of per-view ``HumanMotionIR`` instances into one IR.

    Args:
        irs: list of V ``HumanMotionIR`` objects, one per view.
        cameras: list of V ``Camera`` objects, aligned with ``irs``.
        fusion_module: the fusion backend to use (e.g. ``DLTFusion()``).
        fused_sequence_id: optional sequence id for the output IR.

    Returns:
        A single ``HumanMotionIR`` containing the fused 3D skeleton.

    Raises:
        ValueError: if IRs have mismatched lengths, fps, or human_model.
        NotImplementedError: if per-view 2D observations are not present.
    """
    if len(irs) != len(cameras):
        raise ValueError("Number of IRs must match number of cameras.")
    if len(irs) == 0:
        raise ValueError("At least one IR is required.")

    # Validate consistency across views.
    reference_ir = irs[0]
    T = len(reference_ir.timestamps)
    for idx, ir in enumerate(irs):
        if len(ir.timestamps) != T:
            raise ValueError(f"IR {idx}: frame count mismatch ({len(ir.timestamps)} vs {T}).")
        if ir.fps != reference_ir.fps:
            raise ValueError(f"IR {idx}: fps mismatch ({ir.fps} vs {reference_ir.fps}).")
        if ir.human_model != reference_ir.human_model:
            raise ValueError(f"IR {idx}: human_model mismatch ({ir.human_model} vs {reference_ir.human_model}).")

    # Build per-view 2D observations and confidences.
    views = [ir.sequence_id or f"view_{i}" for i, ir in enumerate(irs)]
    J = None
    points_2d_list = []
    confidence_list = []
    for idx, ir in enumerate(irs):
        if ir.per_view_2d is None or ir.per_view_confidence is None:
            raise NotImplementedError(
                f"IR {idx} ({ir.sequence_id}) does not contain per_view_2d/confidence. "
                "SMPL reprojection is not yet implemented; populate per_view_2d manually."
            )
        view_id = views[idx]
        p2d = ir.per_view_2d[view_id]
        conf = ir.per_view_confidence[view_id]
        if p2d.shape[:2] != (T, J if J is not None else p2d.shape[1]):
            pass  # J will be inferred below
        points_2d_list.append(p2d)
        confidence_list.append(conf)

    points_2d = np.stack(points_2d_list, axis=1)  # (T, V, J, 2)
    confidences = np.stack(confidence_list, axis=1)  # (T, V, J)

    # Run the fusion backend.
    fused_joints_3d = fusion_module.fuse(points_2d, confidences, cameras)  # (T, J, 3)

    # Rebuild the SMPL-ish pose by shifting the reference IR's root translation.
    fused_pose = _align_root(reference_ir, fused_joints_3d)

    # Aggregate per-view betas by simple average.
    fused_betas = _average_betas(irs)
    if fused_betas is not None:
        fused_pose["betas"] = fused_betas

    # Build the fused IR.
    fused_ir = HumanMotionIR(
        schema_version=reference_ir.schema_version,
        sequence_id=fused_sequence_id or f"{reference_ir.sequence_id}_fused",
        person_id=reference_ir.person_id,
        fps=reference_ir.fps,
        timestamps=reference_ir.timestamps.copy(),
        human_model=reference_ir.human_model,
        pose=fused_pose,
        coordinate_system=reference_ir.coordinate_system.copy(),
        views=views,
        camera_parameters={
            view: {"K": cam.K, "R": cam.R, "t": cam.t}
            for view, cam in zip(views, cameras)
        },
        per_view_2d={view: points_2d[:, i, :, :].copy() for i, view in enumerate(views)},
        per_view_confidence={view: confidences[:, i, :].copy() for i, view in enumerate(views)},
        fusion_method=fusion_module.name,
        uncertainty={
            "fused_joints_3d": fused_joints_3d.copy(),
        },
        quality={
            "num_views": len(irs),
            "fusion_method": fusion_module.name,
        },
        provenance={
            **reference_ir.provenance,
            "fused_from": [ir.sequence_id for ir in irs],
            "fusion_module": fusion_module.name,
        },
    )
    return fused_ir


def _align_root(reference_ir: HumanMotionIR, fused_joints_3d: np.ndarray) -> dict:
    """Return a pose dict where the root translation follows the fused skeleton.

    We keep body_pose/global_orient/betas from the reference IR and shift
    ``transl`` by the difference between the fused root and the reference root.
    """
    pose = reference_ir.pose.copy()
    # Prefer explicit transl; if absent, fall back to the first joint of the
    # reference IR (best effort).
    if "transl" in pose:
        reference_root = pose["transl"].copy()  # (T, 3)
    else:
        # Use the pelvis/root joint if we can guess it; otherwise zero.
        reference_root = np.zeros_like(fused_joints_3d[:, 0, :])  # (T, 3)

    fused_root = fused_joints_3d[:, 0, :]  # (T, 3)
    shift = fused_root - reference_root
    pose["transl"] = reference_root + shift
    return pose


def _average_betas(irs: List[HumanMotionIR]) -> np.ndarray | None:
    """Average per-view betas if available."""
    betas = [ir.pose.get("betas") for ir in irs if ir.pose.get("betas") is not None]
    if not betas:
        return None
    # Normalize to first frame shape if time-varying.
    normalized = []
    for b in betas:
        if b.ndim == 2 and b.shape[0] > 1:
            # (T, B) -> average over time -> (B,)
            normalized.append(b.mean(axis=0))
        else:
            normalized.append(b)
    return np.stack(normalized, axis=0).mean(axis=0)
