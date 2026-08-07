"""SMPL-aware fusion module for OmniMultiViewFusion v5 (design_v22).

This module augments :class:`OmniMultiViewFusionV5` with a lightweight SMPL
shape/pose head.  The head consumes the same pooled per-joint feature that
feeds the residual refinement MLP and predicts shared clip-level body shape
(``betas``) plus per-frame pose (``body_pose``, ``global_orient``,
``transl``).  When ``smplx`` is available and a model path is supplied, the
forward pass runs the parametric body and blends its 3D joint locations with
the triangulation-based prediction, producing an SMPL prior for 3D joint
estimation.

The module is intentionally optional with respect to ``smplx``: the head
always emits SMPL parameter predictions, but it only runs the parametric
body when ``smplx`` is installed and ``smpl_model_path`` points to a valid
model.  This lets the file be imported and unit-tested in environments without
``smplx``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5

# Optional smplx dependency.
try:
    import smplx

    HAS_SMPLX = True
except Exception:  # pragma: no cover
    HAS_SMPLX = False


class SMPLPriorHead(nn.Module):
    """Predict SMPL shape/pose parameters and optionally forward the body.

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

        # The input is the concatenation of pooled per-joint features and the
        # raw triangulated 3D joint positions: (N, J, d + 3).
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

        # Learned per-frame blend weight for combining the SMPL prior with the
        # triangulation-based estimate.  The sigmoid keeps the blend factor in
        # (0, 1).
        self.prior_blend = nn.Sequential(
            nn.Linear(d, 1),
            nn.Sigmoid(),
        )

    def _load_smpl(self, batch_size: int, device: torch.device) -> Optional[nn.Module]:
        if not HAS_SMPLX or self.smpl_model_path is None:
            return None
        if self.smpl_model is None or self.smpl_model.batch_size < batch_size:
            self.smpl_model = smplx.SMPL(self.smpl_model_path, batch_size=batch_size).to(device)
        return self.smpl_model

    def forward(self, residual_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the SMPL prior head.

        Args:
            residual_input: (N, J, d + 3) concatenation of pooled per-joint
                feature and raw triangulated 3D joints.

        Returns:
            Dictionary containing ``betas``, ``body_pose``, ``global_orient``,
            ``transl``, ``blend`` and optionally ``smpl_joints`` /
            ``pred_joints_17``.
        """
        N = residual_input.shape[0]
        device = residual_input.device

        x = residual_input.view(N, -1)
        x = self.feat_mlp(x)  # (N, d)

        # Shared clip-level shape: average across the temporal/augmented batch.
        betas = self.betas_head(x).mean(dim=0, keepdim=True)  # (1, 10)
        body_pose = self.body_pose_head(x)  # (N, 69)
        global_orient = self.global_orient_head(x)  # (N, 3)
        transl = self.transl_head(x)  # (N, 3)
        blend = self.prior_blend(x)  # (N, 1)

        out: Dict[str, torch.Tensor] = {
            "betas": betas,
            "body_pose": body_pose,
            "global_orient": global_orient,
            "transl": transl,
            "blend": blend,
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
            # The first 17 SMPL joints approximately correspond to the
            # H36M-style 17-joint skeleton used elsewhere in the repo.
            out["pred_joints_17"] = smpl_out.joints[:, :17, :]  # (N, 17, 3)

        return out


class SMPLPriorFusionV22(OmniMultiViewFusionV5):
    """OmniMultiViewFusion v5 with an SMPL shape/pose prior.

    Inherits the full ``OmniMultiViewFusionV5`` forward pass and intercepts
    the input to the residual MLP to feed the SMPL prior head.  The SMPL
    prior is blended with the triangulation-based 3D estimate using a
    learned per-frame blending weight.

    Parameters
    ----------
    smpl_model_path:
        Optional path to ``SMPL_NEUTRAL.pkl``.  If omitted, the SMPL forward is
        skipped and only raw parameter predictions are returned.
    freeze_base:
        If ``True``, freeze all base-model parameters so only the SMPL prior
        head is trained.  Useful for ablations.
    """

    def __init__(
        self,
        *args,
        smpl_model_path: Optional[str] = None,
        freeze_base: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shape_pose_head = SMPLPriorHead(
            d=self.d,
            n_joints=self.j,
            smpl_model_path=smpl_model_path,
        )
        self._residual_mlp_input: Optional[torch.Tensor] = None
        self._register_hook()

        if freeze_base:
            for param in super().parameters():
                param.requires_grad = False
            for param in self.shape_pose_head.parameters():
                param.requires_grad = True

    def _register_hook(self) -> None:
        """Capture the concatenated (feature, 3D pose) input to ``residual_mlp``."""

        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._residual_mlp_input = inputs[0]

        self.residual_mlp.register_forward_hook(hook)

    def forward(
        self,
        x: torch.Tensor,
        cameras: Optional[List[object]] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        return_smpl: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Returns:
            By default returns the same tuple as ``OmniMultiViewFusionV5``:
            ``(pred_3d, weights, visibility, L, epi_loss, ...)``.
            If ``return_smpl`` is ``True``, the final element of the tuple is
            the SMPL output dictionary (may be ``None`` if the head did not
            produce joints).
        """
        # Run base v5 forward; the hook stores the residual MLP input.
        self._residual_mlp_input = None
        base_out = super().forward(
            x,
            cameras=cameras,
            K=K,
            R=R,
            t=t,
            view_mask=view_mask,
            domain_id=domain_id,
        )

        pred_3d = base_out[0]

        smpl_out: Optional[Dict[str, torch.Tensor]] = None
        if self._residual_mlp_input is not None:
            smpl_out = self.shape_pose_head(self._residual_mlp_input)

            if smpl_out is not None and "pred_joints_17" in smpl_out:
                # Blend the SMPL prior with the triangulation-based estimate.
                B, T, J, _ = pred_3d.shape
                smpl_joints = smpl_out["pred_joints_17"].view(B, T, J, 3)
                blend = smpl_out["blend"].view(B, T, 1, 1)
                pred_3d = (1.0 - blend) * pred_3d + blend * smpl_joints

        if return_smpl:
            return (*base_out, smpl_out)
        return base_out
