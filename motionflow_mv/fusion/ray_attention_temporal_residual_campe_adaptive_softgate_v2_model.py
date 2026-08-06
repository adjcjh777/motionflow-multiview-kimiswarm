"""Temporal ray-aware fusion with camera PE, residual refinement and an improved
adaptive soft view gate (v2).

The v1 soft gate (``RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGate``)
applies a per-view, per-joint MLP to the encoder tokens and thresholds the score
with a sigmoid.  This v2 variant makes three improvements:

1. **Cross-view attention scores.** Views attend to each other before scoring,
   so the gate can compare geometric quality and confidence across the whole rig.
2. **Confidence and ray-geometry aware.** The gate receives the raw confidence
   and the world-space ray direction, letting it reason about occlusion and
   baseline quality.
3. **Learnable temperature.** A scalar ``gate_tau`` controls sigmoid sharpness
   and is learned end-to-end.

The gate is still continuous (soft), so triangulation remains differentiable and
no straight-through estimator is required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ray_attention_temporal_residual_campe_model import RayAttentionFusionModelTemporalResidualCamPE
from .ray_attention_model import _compute_rays, _triangulate_weighted_dlt


class SoftViewGateV2(nn.Module):
    """Cross-view attention soft gate for per-view, per-joint selection.

    Parameters
    ----------
    d:
        Encoder feature dimension.
    n_views:
        Number of views.
    n_heads:
        Number of attention heads for the cross-view score attention.
    target_k:
        Desired average number of active views per joint (used only by the
        regulariser).
    min_views:
        Minimum number of views that must survive the gate; enforced via a
        soft ReLU penalty.
    lambda_gate:
        Weight of the gate regulariser in the loss.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        n_heads: int = 4,
        target_k: int = 4,
        min_views: int = 2,
        lambda_gate: float = 0.01,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads
        self.target_k = target_k
        self.min_views = min_views
        self.lambda_gate = lambda_gate

        # Per-view scoring token: feature + confidence + world ray direction.
        self.score_dim = d + 4

        # Cross-view attention lets views "see" each other before scoring.
        self.view_attn = nn.MultiheadAttention(
            embed_dim=self.score_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.score_dim)

        # Small per-view score MLP.
        self.score_mlp = nn.Sequential(
            nn.Linear(self.score_dim, d),
            nn.ReLU(),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

        # Learnable temperature (log-space for stability).
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        feat: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        confidences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft gate and regularisation loss.

        Args:
            feat: (N, V, J, d) encoder tokens.
            points_2d: (N, V, J, 2) input 2D keypoints (pixels).
            K: (N, V, 3, 3) intrinsics.
            R: (N, V, 3, 3) rotations.
            t: (N, V, 3) translation.
            confidences: (N, V, J) input confidences.

        Returns:
            gate: (N, V, J) soft gate in [0, 1].
            reg: scalar regularisation loss.
        """
        N, V, J, d = feat.shape

        # World-space ray directions (already normalised by _compute_rays).
        rays = _compute_rays(points_2d, K, R, t)  # (N, V, J, 3)

        # Build per-view score tokens: [feature, confidence, ray].
        conf = confidences.unsqueeze(-1)  # (N, V, J, 1)
        score_tokens = torch.cat([feat, conf, rays], dim=-1)  # (N, V, J, d+4)

        # Cross-view attention over views, independently per joint.
        # Reshape: (N*J, V, score_dim)
        tokens = score_tokens.permute(0, 2, 1, 3).reshape(N * J, V, self.score_dim)
        attn_out, _ = self.view_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)
        # Restore shape (N, V, J, score_dim).
        tokens = tokens.view(N, J, V, self.score_dim).permute(0, 2, 1, 3)

        # Per-view score and temperature-controlled sigmoid gate.
        scores = self.score_mlp(tokens).squeeze(-1)  # (N, V, J)
        tau = F.softplus(self.log_tau) + 1e-3
        gate = torch.sigmoid(scores / tau)  # (N, V, J)

        # Regularisation.
        # 1. Budget: encourage average active views close to target_k.
        avg_active = gate.mean(dim=1)  # (N, J)
        target = self.target_k / float(V)
        reg_budget = ((avg_active - target) ** 2).mean()

        # 2. Minimum views: penalise selections below min_views.
        sum_active = gate.sum(dim=1)  # (N, J)
        reg_min = torch.relu(float(self.min_views) - sum_active).mean()

        # 3. Entropy: encourage sharper (but not prematurely peaked) decisions.
        # Minimising negative entropy (i.e. maximising entropy) would make gates
        # uniform, which is bad. We instead add a small penalty on the variance
        # of the gate values across views; this pushes scores apart.
        reg_sharp = -gate.std(dim=1).mean()

        reg = reg_budget + reg_min + 0.1 * reg_sharp
        return gate, reg


class RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGateV2(RayAttentionFusionModelTemporalResidualCamPE):
    """CamPE residual model with an improved adaptive soft view gate (v2).

    Parameters
    ----------
    target_k:
        Desired average number of active views per joint (used only by the
        regulariser, not a hard constraint).
    min_views:
        Minimum number of views that must survive the gate; enforced via a
        soft ReLU penalty.
    lambda_gate:
        Weight of the gate regulariser in the loss.
    gate_n_heads:
        Number of attention heads in the cross-view gate scorer.
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
        n_bands: int = 4,
        target_k: int = 4,
        min_views: int = 2,
        lambda_gate: float = 0.01,
        gate_n_heads: int = 4,
        **kwargs,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            n_bands=n_bands,
        )
        self.target_k = target_k
        self.min_views = min_views
        self.lambda_gate = lambda_gate

        self.gate = SoftViewGateV2(
            d=d,
            n_views=n_views,
            n_heads=gate_n_heads,
            target_k=target_k,
            min_views=min_views,
            lambda_gate=lambda_gate,
        )

    def forward(self, x, cameras=None, K=None, R=None, t=None, n_iter: int = 1):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device


        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_model import cameras_to_tensors
            K, R, t = cameras_to_tensors(cameras, device)

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

        feat = self._extract_frame_features(x_flat, K, R, t)

        # Temporal attention.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Weight head.
        feat_for_weight = feat.permute(0, 2, 1, 3)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)
        weights = weights * confidences

        # Improved adaptive soft gate.
        gate, reg = self.gate(feat, points_2d, K, R, t, confidences)
        weights = weights * gate

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        feat_pooled = feat.mean(dim=1)
        pred_3d = pred_3d_raw
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
            delta = self.residual_mlp(residual_input)
            pred_3d = pred_3d + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        gate = gate.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            gate = gate.squeeze(1)

        return pred_3d, weights, gate, reg


if __name__ == "__main__":
    # Quick shape/gradient sanity check.
    from .ray_attention_temporal_residual_model import _make_cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGateV2(
        j=J, d=64, n_views=V
    )
    pred, weights, gate, reg = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert gate.shape == (B, T, V, J)
    assert reg.numel() == 1
    loss = pred.mean() + reg
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("adaptive soft gate v2 sanity check passed")
