"""Domain-adaptive wrapper for synthetic-to-real transfer.

Wraps ``RayAttentionFusionModelTemporalResidual`` and adds:

1. A gradient-reversal (GRL) domain discriminator operating on pooled temporal
   features.  The discriminator tries to tell synthetic (domain=0) and real
   (domain=1) clips apart; the GRL makes the shared encoder produce
   domain-invariant features.

2. Optional domain-specific FiLM-like affine adapters on the temporal features
   before the weight/residual heads, giving the model capacity to handle
   domain-specific appearance while keeping most parameters shared.

3. A lightweight maximum-mean-discrepancy (MMD) helper that can be used to align
   synthetic and real feature distributions when both sets carry 3D labels.

The wrapper is intentionally self-contained: it subclasses
``RayAttentionFusionModelTemporalResidual`` and only overrides ``forward`` to
insert the domain branches.  No changes to the base model or other swarm
members' files are required.

Input signatures match the base model, with the addition of an optional
``domain_labels`` tensor (B,) of ints in {0, 1}.  If omitted, all samples are
assumed to be from the source (synthetic) domain.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from ..calibration.camera import Camera


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
    """Gradient reversal layer wrapper with scalar lambda."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainAdaptationWrapper(nn.Module):
    """Domain-adaptive wrapper around a temporal-residual backbone.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_temporal_layers, max_temporal_len,
    residual_hidden, use_reproj_gate:
        Passed through to ``RayAttentionFusionModelTemporalResidual``.
    use_domain_classifier:
        If True (default), attach a GRL-based binary domain classifier.
    use_domain_film:
        If True (default), attach domain-specific affine adapters on the
        temporal features before the prediction heads.
    grl_lambda:
        Scaling factor applied inside the gradient reversal layer.  Larger
        values enforce stronger domain invariance.

    Inputs
    ------
    Same as ``RayAttentionFusionModelTemporalResidual`` plus an optional
    ``domain_labels`` tensor of shape (B,) with values in {0, 1}, where 0 is the
    synthetic/source domain and 1 is the real/target domain.  If omitted, all
    samples are treated as source-domain.

    Outputs
    -------
    pred_3d: (B, T, J, 3) or (B, J, 3)
    weights: (B, T, V, J) or (B, V, J)
    domain_logits: (B, 2) if ``use_domain_classifier=True`` and
        ``domain_labels`` is provided or ``return_domain_logits=True``.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        use_reproj_gate: bool = False,
        use_domain_classifier: bool = True,
        use_domain_film: bool = True,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.backbone = RayAttentionFusionModelTemporalResidual(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            use_reproj_gate=use_reproj_gate,
        )
        self.j = j
        self.d = d
        self.n_views = n_views
        self.use_domain_classifier = use_domain_classifier
        self.use_domain_film = use_domain_film

        # Domain discriminator on pooled temporal features.
        if self.use_domain_classifier:
            self.grl = GradientReversalLayer(lambda_=grl_lambda)
            self.domain_classifier = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, 2),
            )

        # Domain-specific FiLM: one affine per domain, applied to temporal feats.
        if self.use_domain_film:
            self.domain_film = nn.ModuleDict(
                {
                    "0": nn.Linear(d, d * 2),
                    "1": nn.Linear(d, d * 2),
                }
            )
            # Learnable domain embedding used to predict the affine parameters.
            self.domain_embed = nn.Embedding(2, d)

    def _apply_domain_film(self, feat: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """Apply per-domain affine (FiLM) modulation to (B*T, V, J, d) features.

        Args:
            feat: (N, V, J, d) temporal features where N = B*T.
            domain_labels: (B,) domain ids expanded to (N,).

        Returns:
            modulated features of shape (N, V, J, d).
        """
        N, V, J, d = feat.shape
        # Per-sample domain embedding, broadcast across views/joints.
        emb = self.domain_embed(domain_labels)  # (N, d)
        params = self.domain_film["0"](emb) * (domain_labels.unsqueeze(-1) == 0).float()
        params = params + self.domain_film["1"](emb) * (domain_labels.unsqueeze(-1) == 1).float()
        gamma, beta = params.chunk(2, dim=-1)  # each (N, d)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return feat * (1.0 + gamma) + beta

    def _domain_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute binary domain logits from pooled temporal features.

        Args:
            feat: (N, V, J, d) temporal features.

        Returns:
            domain_logits: (N, 2).
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
        n_iter: int = 1,
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

        # Per-frame v3 features.
        feat = self.backbone._extract_frame_features(x_flat, K, R, t)

        # Temporal transformer.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.backbone.temporal_pos_embed[:T]
        for layer in self.backbone.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Domain discriminator on pooled features (before optional FiLM).
        domain_logits = None
        if self.use_domain_classifier and (domain_labels is not None or return_domain_logits):
            domain_logits = self._domain_logits(feat)

        # Optional domain-specific FiLM modulation.
        if self.use_domain_film and domain_labels is not None:
            # Expand domain labels from (B,) to (B*T,) by repeating over T.
            if domain_labels.numel() == B:
                domain_labels_expanded = domain_labels.unsqueeze(1).expand(B, T).reshape(-1)
            else:
                domain_labels_expanded = domain_labels.reshape(-1)
            feat = self._apply_domain_film(feat, domain_labels_expanded)

        # Per-frame weight prediction and DLT triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)
        w_logits = self.backbone.weight_head(feat_for_weight).squeeze(-1)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)
        weights = weights * confidences

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = self.backbone._triangulate(points_2d, weights, P, K, R, t)

        # Residual refinement head (with optional reprojection gate).
        feat_pooled = feat.mean(dim=1)
        pred_3d = pred_3d_raw
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
            delta = self.backbone.residual_mlp(residual_input)
            if self.backbone.use_reproj_gate:
                summary = self.backbone._reprojection_error_summary(pred_3d, points_2d, P, inlier_thresh=10.0)
                gate_input = torch.cat([residual_input, summary], dim=-1)
                gate = self.backbone.reproj_gate(gate_input)
                delta = gate * delta
            pred_3d = pred_3d + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        if self.use_domain_classifier and (domain_labels is not None or return_domain_logits):
            return pred_3d, weights, domain_logits
        return pred_3d, weights


def maximum_mean_discrepancy(feat_src: torch.Tensor, feat_tgt: torch.Tensor, kernel: str = "rbf") -> torch.Tensor:
    """Compute MMD^2 between source and target feature sets.

    Args:
        feat_src: (N_src, D) source features.
        feat_tgt: (N_tgt, D) target features.
        kernel: only "rbf" is supported.

    Returns:
        MMD^2 scalar tensor.
    """
    if kernel != "rbf":
        raise NotImplementedError("Only RBF kernel is supported for MMD")

    # Use a fixed Gaussian bandwidth for simplicity.
    bandwidth = 1.0
    xx = torch.mm(feat_src, feat_src.t())
    yy = torch.mm(feat_tgt, feat_tgt.t())
    xy = torch.mm(feat_src, feat_tgt.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    rxy = xx.diag().unsqueeze(1).expand_as(xy)
    ryx = yy.diag().unsqueeze(0).expand_as(xy)

    k_xx = torch.exp(-(rx + rx.t() - 2 * xx) / (2 * bandwidth ** 2))
    k_yy = torch.exp(-(ry + ry.t() - 2 * yy) / (2 * bandwidth ** 2))
    k_xy = torch.exp(-(rxy + ryx - 2 * xy) / (2 * bandwidth ** 2))

    m = k_xx.size(0)
    n = k_yy.size(0)
    mmd = k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2 * k_xy.sum() / (m * n)
    return mmd


if __name__ == "__main__":
    # Quick shape/gradient sanity check.
    B, T, V, J = 2, 5, 4, 17
    x = torch.rand(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
    t = torch.randn(B, V, 3)

    model = DomainAdaptationWrapper(j=J, d=64, n_views=V)
    domain_labels = torch.zeros(B, dtype=torch.long)
    pred, w, dlogits = model(x, K=K, R=R, t=t, domain_labels=domain_labels, return_domain_logits=True)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    assert dlogits.shape == (B * T, 2)
    loss = pred.mean() + dlogits.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("domain adaptation wrapper sanity check passed")
