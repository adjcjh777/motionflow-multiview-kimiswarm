"""Domain-adaptive wrapper around the current best PP model.

Wraps ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` and adds:

1. A gradient-reversal (GRL) domain discriminator operating on pooled spatio-
   temporal features.  The discriminator tries to tell source (domain=0) and
   target (domain=1) clips apart; the GRL makes the shared encoder produce
   domain-invariant features.

2. Optional domain-specific FiLM-like affine adapters on the spatio-temporal
   features before the weight head, giving the model capacity to handle
   domain-specific appearance while keeping most parameters shared.

The wrapper is intentionally self-contained: it composes the backbone and only
re-implements ``forward`` to insert the domain branches.  No changes to the
base model or other swarm members' files are required.

Input signatures match the backbone, with the addition of an optional
``domain_labels`` tensor ``(B,)`` of ints in ``{0, 1}``.  If omitted, all samples
are assumed to be from the source (synthetic) domain and FiLM is bypassed;
GRL logits can still be returned via ``return_domain_logits=True``.

Output
------
- ``pred_3d``: refined 3D pose, same shape as the backbone.
- ``weights``: predicted per-view weights, same shape as the backbone.
- ``domain_logits``: ``(B*T, 2)`` if ``use_domain_classifier=True`` and either
  ``domain_labels`` is provided or ``return_domain_logits=True``.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..calibration.camera import Camera
from ..fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from ..fusion.ray_attention_model import _triangulate_weighted_dlt


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient-reversal layer with a scalar lambda."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainAdaptationWrapper(nn.Module):
    """Domain-adaptive wrapper around the principal-point residual model.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_st_layers, max_temporal_len,
    residual_hidden, principal_point_hidden, principal_point_max_offset,
    focal_max_scale:
        Passed through to the PP backbone.
    use_domain_classifier:
        If True (default), attach a GRL-based binary domain classifier.
    use_domain_film:
        If True (default), attach domain-specific affine adapters on the
        spatio-temporal features.
    grl_lambda:
        Scaling factor inside the gradient-reversal layer.
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
        use_domain_classifier: bool = True,
        use_domain_film: bool = True,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.backbone = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
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
        )
        self.j = j
        self.d = d
        self.n_views = n_views
        self.use_domain_classifier = use_domain_classifier
        self.use_domain_film = use_domain_film

        if self.use_domain_classifier:
            self.grl = GradientReversalLayer(lambda_=grl_lambda)
            self.domain_classifier = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, 2),
            )

        if self.use_domain_film:
            self.domain_film = nn.ModuleDict(
                {
                    "0": nn.Linear(d, d * 2),
                    "1": nn.Linear(d, d * 2),
                }
            )
            self.domain_embed = nn.Embedding(2, d)

    def _apply_domain_film(
        self, feat: torch.Tensor, domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-domain affine (FiLM) modulation to ``(N, V, J, d)`` features.

        Args:
            feat: ``(N, V, J, d)`` spatio-temporal features.
            domain_labels: ``(N,)`` domain ids in ``{0, 1}``.

        Returns:
            Modulated features of shape ``(N, V, J, d)``.
        """
        N, V, J, d = feat.shape
        emb = self.domain_embed(domain_labels)  # (N, d)
        params = self.domain_film["0"](emb) * (domain_labels.unsqueeze(-1) == 0).float()
        params = params + self.domain_film["1"](emb) * (domain_labels.unsqueeze(-1) == 1).float()
        gamma, beta = params.chunk(2, dim=-1)  # each (N, d)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return feat * (1.0 + gamma) + beta

    def _domain_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute binary domain logits from pooled spatio-temporal features.

        Args:
            feat: ``(N, V, J, d)`` features.

        Returns:
            ``(N, 2)`` domain logits.
        """
        pooled = feat.mean(dim=(1, 2))  # (N, d)
        return self.domain_classifier(self.grl(pooled))

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        domain_labels: torch.Tensor = None,
        return_domain_logits: bool = False,
    ):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            K, R, t = _cameras_to_tensors(cameras, device)

        # Broadcast camera tensors to (B*T, V, ...).
        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.backbone.principal_point_correction(
            K=K, x=x_flat, weights=confidences
        )
        K_corrected = correction_outputs[0]
        focal_scale = correction_outputs[2] if self.backbone.correct_focal else None

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self.backbone._extract_frame_features(x_flat, K_corrected, R, t)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.backbone.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.backbone.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.backbone.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Domain discriminator on pooled features (before optional FiLM).
        domain_logits = None
        if self.use_domain_classifier and (domain_labels is not None or return_domain_logits):
            domain_logits = self._domain_logits(feat)

        # Optional domain-specific FiLM modulation.
        if self.use_domain_film and domain_labels is not None:
            if domain_labels.numel() == B:
                domain_labels_expanded = domain_labels.unsqueeze(1).expand(B, T).reshape(-1)
            else:
                domain_labels_expanded = domain_labels.reshape(-1)
            feat = self._apply_domain_film(feat, domain_labels_expanded)

        # Optional visibility-aware weighting (base returns 1).
        visibility = self.backbone._visibility_multiplier(feat, confidences)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.backbone.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)
        delta = self.backbone.residual_mlp(residual_input)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        if self.use_domain_classifier and (domain_labels is not None or return_domain_logits):
            return pred_3d, weights, domain_logits
        return pred_3d, weights
