"""Multi-task shape / pose head for ray-aware temporal fusion.

Adds a lightweight SMPL shape/pose branch on top of the residual refinement
head in ``RayAttentionFusionModelTemporalResidual``.  The new head consumes the
same per-joint pooled feature that drives the residual MLP and predicts a
shared clip-level ``betas`` vector plus per-frame ``body_pose``,
``global_orient`` and ``transl``.

The implementation is intentionally optional with respect to ``smplx``: the
head always emits SMPL parameter predictions, but it only runs the parametric
body when ``smplx`` is installed and a model path is supplied.  This lets the
file be imported and unit-tested in environments without ``smplx``.

Input / Output
--------------
The wrapper ``MultiTaskShapePoseModel`` follows the same contract as
``RayAttentionFusionModelTemporalResidual`` (input ``(B, T, V, J, 3)`` or
``(B, V, J, 3)``) and additionally returns SMPL parameters when
``return_smpl=True``.

References
----------
- docs/swarm_iter_next/design_multi_task_shape_pose/report.md
- motionflow_mv/fusion/ray_attention_temporal_residual_model.py
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Optional smplx dependency.
try:
    import smplx

    HAS_SMPLX = True
except Exception:  # pragma: no cover
    HAS_SMPLX = False

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


class MultiTaskShapePoseHead(nn.Module):
    """Predict shared clip-level betas and per-frame SMPL pose/transl.

    Parameters
    ----------
    d:
        Per-joint feature dimension produced by the ray-attention encoder.
    n_joints:
        Number of joints ``J`` in the skeleton (default 17 for H36M-style).
    smpl_model_path:
        Optional path to ``SMPL_NEUTRAL.pkl``.  If ``None`` or ``smplx`` is not
        installed, the head returns parameter predictions only.
    """

    def __init__(
        self,
        d: int,
        n_joints: int = 17,
        smpl_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.d = d
        self.n_joints = n_joints
        self.smpl_model_path = smpl_model_path
        self.smpl_model: Optional[nn.Module] = None

        # The input to the residual MLP is (N, J, d + 3) where the last three
        # channels are the raw triangulated 3D joint positions.  We pool over
        # joints with a small MLP.
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_joints * (d + 3), d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )

        # Shared clip-level shape.
        self.betas_head = nn.Linear(d, 10)
        # Per-frame pose and translation.
        self.body_pose_head = nn.Linear(d, 69)
        self.global_orient_head = nn.Linear(d, 3)
        self.transl_head = nn.Linear(d, 3)

    def _load_smpl(self, batch_size: int, device: torch.device) -> Optional[nn.Module]:
        if not HAS_SMPLX or self.smpl_model_path is None:
            return None
        if self.smpl_model is None or self.smpl_model.batch_size < batch_size:
            self.smpl_model = smplx.SMPL(self.smpl_model_path, batch_size=batch_size).to(device)
        return self.smpl_model

    def forward(self, residual_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the shape/pose head.

        Args:
            residual_input: (N, J, d + 3) concatenation of pooled per-joint
                feature and raw triangulated 3D joints.

        Returns:
            Dictionary containing ``betas``, ``body_pose``, ``global_orient``,
            ``transl`` and optionally ``smpl_joints`` / ``pred_joints_17``.
        """
        N = residual_input.shape[0]
        device = residual_input.device

        # Flatten per-joint tokens and reduce to a compact vector.
        x = residual_input.view(N, -1)
        x = self.feat_mlp(x)  # (N, d)

        # Shared betas: average across the temporal/augmented batch.
        betas = self.betas_head(x).mean(dim=0, keepdim=True)  # (1, 10)
        body_pose = self.body_pose_head(x)  # (N, 69)
        global_orient = self.global_orient_head(x)  # (N, 3)
        transl = self.transl_head(x)  # (N, 3)

        out: Dict[str, torch.Tensor] = {
            "betas": betas,
            "body_pose": body_pose,
            "global_orient": global_orient,
            "transl": transl,
        }

        smpl_model = self._load_smpl(N, device)
        if smpl_model is not None:
            smpl_out = smpl_model(
                betas=betas.expand(N, -1),
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
            out["smpl_joints"] = smpl_out.joints  # (N, 24 or more, 3)
            # First 17 SMPL joints approximately correspond to the H36M-style
            # skeleton used by the rest of the repository.
            out["pred_joints_17"] = smpl_out.joints[:, :17, :]  # (N, 17, 3)

        return out


class MultiTaskShapePoseModel(RayAttentionFusionModelTemporalResidual):
    """Temporal residual ray-attention fusion with a multi-task SMPL head.

    Inherits the full ``RayAttentionFusionModelTemporalResidual`` forward pass
    and intercepts the input to the residual MLP to feed the shape/pose head.
    This avoids duplicating the parent forward and keeps the prototype
    self-contained in one file.

    Parameters
    ----------
    smpl_model_path:
        Optional path to ``SMPL_NEUTRAL.pkl``.  If omitted, the SMPL forward is
        skipped and only raw parameter predictions are returned.
    """

    def __init__(
        self,
        *args,
        smpl_model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shape_pose_head = MultiTaskShapePoseHead(
            d=self.d,
            n_joints=self.j,
            smpl_model_path=smpl_model_path,
        )
        self._residual_mlp_input: Optional[torch.Tensor] = None
        self._register_hook()

    def _register_hook(self) -> None:
        """Capture the concatenated (feature, 3D pose) input to ``residual_mlp``."""

        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._residual_mlp_input = inputs[0]

        self.residual_mlp.register_forward_hook(hook)

    def forward(
        self,
        x: torch.Tensor,
        cameras=None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        n_iter: int = 1,
        return_smpl: bool = False,
    ):
        """Forward pass.

        Returns:
            If ``return_smpl`` is False: ``(pred_3d, weights)``.
            If ``return_smpl`` is True: ``(pred_3d, weights, smpl_out)`` where
            ``smpl_out`` may be ``None`` if the head produced no output.
        """
        pred_3d, weights = super().forward(x, cameras=cameras, K=K, R=R, t=t, n_iter=n_iter)


        smpl_out: Optional[Dict[str, torch.Tensor]] = None
        if return_smpl and self._residual_mlp_input is not None:
            smpl_out = self.shape_pose_head(self._residual_mlp_input)

        if return_smpl:
            return pred_3d, weights, smpl_out
        return pred_3d, weights


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for tests)."""
    import numpy as np
    from ..calibration.camera import Camera

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


if __name__ == "__main__":
    # Smoke test: forward, backward, and optional SMPL output shapes.
    import numpy as np

    B, T, V, J = 2, 5, 4, 17
    d = 64
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    y_3d = torch.randn(B, T, J, 3)

    model = MultiTaskShapePoseModel(j=J, d=d, n_views=V)
    pred_3d, weights, smpl_out = model(x, cameras=cameras, return_smpl=True)

    assert pred_3d.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert smpl_out is not None
    assert smpl_out["betas"].shape == (1, 10)
    assert smpl_out["body_pose"].shape == (B * T, 69)
    assert smpl_out["global_orient"].shape == (B * T, 3)
    assert smpl_out["transl"].shape == (B * T, 3)

    loss = (pred_3d - y_3d).pow(2).mean()
    if "pred_joints_17" in smpl_out:
        assert smpl_out["pred_joints_17"].shape == (B * T, 17, 3)
        loss = loss + (smpl_out["pred_joints_17"].mean() * 0.0)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    print(f"MultiTaskShapePoseModel smoke test passed (smplx available={HAS_SMPLX}).")
