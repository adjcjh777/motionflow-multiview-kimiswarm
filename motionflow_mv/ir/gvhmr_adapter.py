"""Adapter between GVHMR hmr4d_results.pt and HumanMotionIR.

GVHMR output keys (from `tools/demo/demo.py` predict step):
    pred = {
        "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
        "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
        "K_fullimg": data["K_fullimg"],
        "net_outputs": outputs,
    }

This adapter converts the global SMPL params into HumanMotionIR and back.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .human_motion_ir import HumanMotionIR


SMPL_POSE_KEYS = ("body_pose", "global_orient", "transl", "betas")


def gvhmr_pt_to_ir(pt_path: str | Path, sequence_id: str = "") -> HumanMotionIR:
    """Load a GVHMR hmr4d_results.pt and convert to HumanMotionIR."""
    pt_path = Path(pt_path)
    pred = torch.load(pt_path, map_location="cpu")
    if "smpl_params_global" not in pred:
        raise ValueError(f"{pt_path} does not contain 'smpl_params_global'")

    smpl = pred["smpl_params_global"]
    # Determine number of frames from first available pose key
    first_key = next(k for k in SMPL_POSE_KEYS if k in smpl)
    T = smpl[first_key].shape[0]
    fps = 30.0  # GVHMR output is resampled to 30 fps
    timestamps = np.arange(T, dtype=np.float32) / fps

    pose = {}
    for key in SMPL_POSE_KEYS:
        if key in smpl:
            v = smpl[key]
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            pose[key] = v

    # Camera intrinsics are kept in provenance for reconstruction.
    K_fullimg = pred.get("K_fullimg")
    if isinstance(K_fullimg, torch.Tensor):
        K_fullimg = K_fullimg.numpy()

    coordinate_system = {
        "handedness": "right",
        "up_axis": "y",
        "forward_axis": "z",
        "length_unit": "meter",
        "world_from_reference": np.eye(4, dtype=np.float32),
    }

    provenance = {
        "source_format": "GVHMR hmr4d_results.pt",
        "source_path": str(pt_path.resolve()),
        "K_fullimg": K_fullimg,
    }

    return HumanMotionIR(
        schema_version="0.1.0",
        sequence_id=sequence_id or pt_path.stem,
        person_id="person_0",
        fps=fps,
        timestamps=timestamps,
        human_model="smplx" if "smplx" in str(pt_path).lower() else "smpl",
        pose=pose,
        coordinate_system=coordinate_system,
        provenance=provenance,
    )


def ir_to_gvhmr_pt(ir: HumanMotionIR) -> Dict[str, Any]:
    """Convert HumanMotionIR back to a GVHMR-compatible dict.

    The returned dict contains the minimal keys needed by downstream GMR:
        - smpl_params_global
        - smpl_params_incam (copy of global as a fallback)
        - K_fullimg (identity if unknown)
    """
    smpl_params = {}
    for key in SMPL_POSE_KEYS:
        v = ir.pose.get(key)
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        smpl_params[key] = v

    K_fullimg = ir.provenance.get("K_fullimg")
    if K_fullimg is None:
        K_fullimg = torch.eye(3).unsqueeze(0).repeat(len(ir.timestamps), 1, 1)
    elif isinstance(K_fullimg, np.ndarray):
        K_fullimg = torch.from_numpy(K_fullimg)

    return {
        "smpl_params_global": smpl_params,
        "smpl_params_incam": {k: v.clone() for k, v in smpl_params.items()},
        "K_fullimg": K_fullimg,
    }
