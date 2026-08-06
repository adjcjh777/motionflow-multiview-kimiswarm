"""MotionFlow MultiViewFusionPlugin: drop-in pipeline integration.

This module provides a high-level plugin that wires the low-level
``FusionModule`` backends (DLT, attention, temporal-residual, etc.) into the
MotionFlow single-view pipeline.  It is the canonical integration point between
multi-view fusion research code and production inference:

    from motionflow_mv.pipeline_multiview_plugin import MultiViewFusionPlugin
    from motionflow_mv.fusion.fusion_module import FUSION_REGISTRY

    plugin = MultiViewFusionPlugin(fusion_name="ray_attention_temporal_residual")
    fused_ir = plugin.fuse_irs(per_view_irs, cameras)

The plugin is intentionally thin: it normalizes camera units, validates
observations, dispatches to a registered ``FusionModule``, and repackages the
output as a ``HumanMotionIR`` so downstream stages (visualization, retargeting,
robot policy) do not care whether the 3D pose came from a geometric baseline or
a learned model.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .calibration.camera import Camera
from .fusion.fusion_module import FUSION_REGISTRY, FusionModule
from .ir.human_motion_ir import HumanMotionIR
from .ir.multiview_adapter import fuse_multiple_irs


class MultiViewFusionPlugin:
    """Drop-in multi-view fusion plugin for the MotionFlow pipeline.

    Parameters
    ----------
    fusion_name:
        Name of a registered ``FusionModule`` (e.g. ``"dlt"``,
        ``"attention"``, ``"ray_attention_temporal_residual"``).
    fusion_module:
        An already-initialized ``FusionModule`` instance.  If provided,
        ``fusion_name`` is ignored.
    input_scale:
        Scale factor that converts the input camera units into meters.
        ``cam.t`` is divided by this value before fusion; the returned 3D
        joints are always in meters.
    device:
        PyTorch device for learned backends.  If ``None`` the plugin uses
        ``cuda`` when available, otherwise ``cpu``.

    Attributes
    ----------
    fusion : FusionModule
        The wrapped fusion backend.
    """

    def __init__(
        self,
        fusion_name: str = "ray_attention_temporal_residual",
        fusion_module: Optional[FusionModule] = None,
        input_scale: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if fusion_module is not None:
            self.fusion = fusion_module
            fusion_name = fusion_module.name
        else:
            self.fusion = FUSION_REGISTRY.get(fusion_name)

        self.fusion_name = fusion_name
        self.input_scale = input_scale

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move learned backends to the requested device if they expose one.
        if hasattr(self.fusion, "model") and isinstance(self.fusion.model, torch.nn.Module):
            self.fusion.model = self.fusion.model.to(self.device)

    def _normalize_cameras(self, cameras: List[Camera]) -> List[Camera]:
        """Return cameras scaled to meters if ``input_scale != 1.0``."""
        if self.input_scale == 1.0:
            return cameras
        return [
            Camera(
                K=cam.K.copy(),
                R=cam.R.copy(),
                t=cam.t.copy() / self.input_scale,
            )
            for cam in cameras
        ]

    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        """Fuse per-view 2D keypoints into 3D joints.

        Args:
            points_2d: ``(T, V, J, 2)`` or ``(V, J, 2)`` array of 2D keypoints.
            confidences: ``(T, V, J)`` or ``(V, J)`` array of confidence scores.
            cameras: list of ``V`` ``Camera`` objects.

        Returns:
            ``(T, J, 3)`` world-coordinate 3D joints (meters).  If a single
            frame is supplied (``V, J, 2``), a leading temporal dimension is
            added automatically.
        """
        cameras = self._normalize_cameras(cameras)
        return self.fusion.fuse(points_2d, confidences, cameras)

    def fuse_irs(
        self,
        irs: List[HumanMotionIR],
        cameras: List[Camera],
        fused_sequence_id: str = "",
    ) -> HumanMotionIR:
        """Fuse per-view ``HumanMotionIR`` instances into a single IR.

        Args:
            irs: list of ``V`` per-view ``HumanMotionIR`` objects.
            cameras: list of ``V`` ``Camera`` objects, aligned with ``irs``.
            fused_sequence_id: optional sequence id for the output IR.

        Returns:
            A fused ``HumanMotionIR`` whose ``pose["transl"]`` is aligned to
            the recovered 3D skeleton root.
        """
        cameras = self._normalize_cameras(cameras)
        return fuse_multiple_irs(irs, cameras, self.fusion, fused_sequence_id)

    def fuse_from_predictions(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        cameras: List[Camera],
        return_ir: bool = False,
        timestamps: Optional[np.ndarray] = None,
        sequence_id: str = "fused",
    ) -> Union[np.ndarray, HumanMotionIR]:
        """Fuse a dictionary of per-view predictions into 3D joints.

        Args:
            predictions: mapping ``view_id -> {"keypoints_2d": (T, J, 2),
            "confidence": (T, J)}``.  ``keypoints_2d`` and ``confidence``
            can also be ``(V, J, 2)`` / ``(V, J)`` for a single frame.
            cameras: list of ``V`` ``Camera`` objects ordered like
            ``predictions``.
            return_ir: if ``True``, wrap the result in a ``HumanMotionIR``.
            timestamps: required when ``return_ir=True``; ``(T,)`` array in
            seconds.
            sequence_id: used only when ``return_ir=True``.

        Returns:
            Either ``(T, J, 3)`` numpy array or a ``HumanMotionIR``.
        """
        view_ids = list(predictions.keys())
        points_2d_list = []
        confidence_list = []
        for vid in view_ids:
            pred = predictions[vid]
            points_2d_list.append(pred["keypoints_2d"])
            confidence_list.append(pred["confidence"])

        points_2d = np.stack(points_2d_list, axis=1)  # (T, V, J, 2)
        confidences = np.stack(confidence_list, axis=1)  # (T, V, J)

        fused = self.fuse(points_2d, confidences, cameras)

        if not return_ir:
            return fused

        if timestamps is None:
            timestamps = np.arange(fused.shape[0], dtype=np.float64) / 30.0

        return HumanMotionIR(
            schema_version="1.0",
            sequence_id=sequence_id,
            person_id="person_0",
            fps=30.0,
            timestamps=timestamps,
            human_model="smpl",
            pose={"transl": fused[:, 0, :].copy(), "joints_3d": fused.copy()},
            coordinate_system={
                "up_axis": "y",
                "forward_axis": "z",
                "length_unit": "m",
                "world_from_reference": np.eye(4),
            },
            views=view_ids,
            camera_parameters={
                vid: {"K": cam.K, "R": cam.R, "t": cam.t}
                for vid, cam in zip(view_ids, cameras)
            },
            per_view_2d={vid: points_2d[:, i, :, :].copy() for i, vid in enumerate(view_ids)},
            per_view_confidence={vid: confidences[:, i, :].copy() for i, vid in enumerate(view_ids)},
            fusion_method=self.fusion_name,
            uncertainty={"fused_joints_3d": fused.copy()},
            quality={"num_views": len(view_ids), "fusion_method": self.fusion_name},
            provenance={"plugin": "MultiViewFusionPlugin", "fusion_name": self.fusion_name},
        )

    @staticmethod
    def available_backends() -> List[str]:
        """Return the list of registered fusion backend names."""
        return FUSION_REGISTRY.names()


def create_multiview_plugin(
    backend: str = "dlt",
    checkpoint_path: Optional[str] = None,
    input_scale: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> MultiViewFusionPlugin:
    """Factory that instantiates a ``MultiViewFusionPlugin`` from a backend name.

    Args:
        backend: registered backend name.  Use
        ``MultiViewFusionPlugin.available_backends()`` to list options.
        checkpoint_path: optional path to a model checkpoint for learned
        backends.
        input_scale: scale factor converting camera units to meters.
        device: target PyTorch device.
        **kwargs: extra arguments forwarded to the backend constructor when
        ``checkpoint_path`` is provided.

    Returns:
        ``MultiViewFusionPlugin`` ready for inference.
    """
    if checkpoint_path is not None:
        backend_cls = type(FUSION_REGISTRY.get(backend))
        fusion_module = backend_cls(checkpoint_path=checkpoint_path, **kwargs)
    else:
        fusion_module = FUSION_REGISTRY.get(backend)

    return MultiViewFusionPlugin(
        fusion_module=fusion_module,
        input_scale=input_scale,
        device=device,
    )
