"""Adapter between ScoreHMR output and HumanMotionIR.

ScoreHMR (CVPR 2024, MIT license) produces camera-relative SMPL/SMPL-X params.
This adapter converts those params into ``HumanMotionIR`` so that the rest of
MotionFlow treats them like any other per-view IR. World coordinates must come
from downstream multi-view fusion, not from ScoreHMR itself.

The adapter intentionally does not import the ``score_hmr`` package; it only
operates on the dict of tensors that ScoreHMR returns.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .human_motion_ir import HumanMotionIR


SMPL_POSE_KEYS = ("body_pose", "global_orient", "transl", "betas")


def scorehmr_result_to_ir(
    pred_smpl_params: Dict[str, torch.Tensor],
    camera_translation: torch.Tensor | None = None,
    sequence_id: str = "",
    fps: float = 30.0,
) -> HumanMotionIR:
    """Convert ScoreHMR SMPL params to ``HumanMotionIR``.

    Args:
        pred_smpl_params: dict with ``global_orient``, ``body_pose``, ``betas``.
            Rotation matrices are expected, e.g. ``global_orient`` (T, 1, 3, 3)
            and ``body_pose`` (T, 23, 3, 3).
        camera_translation: optional camera-relative translation (T, 3).
        sequence_id: identifier for this view.
        fps: frame rate.

    Returns:
        ``HumanMotionIR`` whose ``transl`` is a placeholder; real world
        coordinates require multi-view fusion.
    """
    body_pose = pred_smpl_params["body_pose"]
    global_orient = pred_smpl_params["global_orient"]
    betas = pred_smpl_params["betas"]

    if body_pose.ndim == 3:
        body_pose = body_pose[None]
        global_orient = global_orient[None]
        if betas.ndim == 1:
            betas = betas[None]

    T = body_pose.shape[0]

    if camera_translation is not None:
        transl = camera_translation.detach().cpu().numpy()
    else:
        transl = np.zeros((T, 3), dtype=np.float32)

    pose: Dict[str, np.ndarray] = {
        "global_orient": global_orient.detach().cpu().numpy().reshape(T, 1, 3, 3),
        "body_pose": body_pose.detach().cpu().numpy().reshape(T, 23, 3, 3),
        "betas": betas.detach().cpu().numpy(),
        "transl": transl,
    }

    coordinate_system = {
        "handedness": "right",
        "up_axis": "y",
        "forward_axis": "z",
        "length_unit": "meter",
        "world_from_reference": np.eye(4, dtype=np.float32),
        "note": "ScoreHMR output is camera-relative; world coordinates require multi-view fusion.",
    }

    return HumanMotionIR(
        schema_version="0.1.0",
        sequence_id=sequence_id or "scorehmr_view",
        person_id="person_0",
        fps=fps,
        timestamps=np.arange(T, dtype=np.float32) / fps,
        human_model="smpl",
        pose=pose,
        coordinate_system=coordinate_system,
        provenance={
            "source_format": "ScoreHMR SMPL params",
            "source_path": "",
            "camera_relative": True,
        },
    )


def ir_to_scorehmr_params(ir: HumanMotionIR) -> Dict[str, Any]:
    """Convert ``HumanMotionIR`` back to ScoreHMR-compatible params (best-effort)."""
    pose = ir.pose
    return {
        "global_orient": torch.from_numpy(pose["global_orient"]),
        "body_pose": torch.from_numpy(pose["body_pose"]),
        "betas": torch.from_numpy(pose["betas"]),
        "transl": torch.from_numpy(
            pose.get("transl", np.zeros((len(ir.timestamps), 3)))
        ),
    }
