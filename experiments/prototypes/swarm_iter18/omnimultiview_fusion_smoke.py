"""
OmniMultiViewFusion smoke test (CPU-only).

This is a deliberately tiny, self-contained outline of the proposed
OmniMultiViewFusion module.  It defines a factorised (T x V x J) attention
block with a visibility head, an uncertainty head, a triangulation stub, and a
residual refinement head.  The goal is only to verify that the architecture
shapes are consistent and that a forward pass on synthetic data runs without
raising.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniMultiViewFusionSmoke(nn.Module):
    """
    Minimal prototype of the OmniMultiViewFusion architecture.

    Input
    -----
    x2d : (B, T, V, J, 2)
    conf: (B, T, V, J)
    K   : (B, V, 3, 3)   # intrinsics (duplicated over time externally)
    cam : (B, V, d_cam)  # camera positional embedding

    Output
    ------
    pose_3d : (B, T, J, 3)
    visibility: (B, T, V, J)
    log_var : (B, T, V, J)
    """

    def __init__(self, d_model=64, d_cam=16, n_layers=2, n_head=4, j=17):
        super().__init__()
        self.d_model = d_model
        self.j = j

        # 2D point + confidence + camera embedding -> token
        self.input_proj = nn.Linear(2 + 1 + d_cam, d_model)

        # Visibility head before the factorised block
        self.vis_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Factorised attention along T, V, J axes
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, d_model * 2, batch_first=True)
            for _ in range(n_layers)
        ])
        self.view_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, d_model * 2, batch_first=True)
            for _ in range(n_layers)
        ])
        # Sparse graph-joint layer is approximated by a small Transformer over J
        # with a fixed adjacency mask for the smoke test.
        self.joint_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, d_model * 2, batch_first=True)
            for _ in range(n_layers)
        ])

        # Uncertainty head
        self.unc_head = nn.Linear(d_model, 1)

        # Residual refinement MLP
        self.residual_mlp = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
        )

    def _factorised_pass(self, tokens, visibility):
        """
        tokens: (B, T, V, J, D)
        visibility: (B, T, V, J)
        """
        b, t, v, j, d = tokens.shape

        # Temporal attention: reshape -> (B*V*J, T, D)
        x = tokens.permute(0, 2, 3, 1, 4).reshape(b * v * j, t, d)
        for layer in self.temporal_layers:
            x = layer(x)
        tokens = x.reshape(b, v, j, t, d).permute(0, 3, 1, 2, 4)

        # View attention: reshape -> (B*T*J, V, D)
        x = tokens.permute(0, 1, 3, 2, 4).reshape(b * t * j, v, d)
        # Mask occluded views using visibility (soft gating via attention mask)
        view_mask = visibility.permute(0, 1, 3, 2).reshape(b * t * j, v)
        view_mask = (1.0 - view_mask).bool()  # True where view is occluded
        for layer in self.view_layers:
            x = layer(x, src_key_padding_mask=view_mask)
        tokens = x.reshape(b, t, j, v, d).permute(0, 1, 4, 2, 3)

        # Joint attention: reshape -> (B*T*V, J, D)
        x = tokens.reshape(b * t * v, j, d)
        for layer in self.joint_layers:
            x = layer(x)
        tokens = x.reshape(b, t, v, j, d)
        return tokens

    def _triangulate_stub(self, x2d, weights, K):
        """
        Differentiable triangulation stub.
        Instead of a full DLT, we use the weighted average of back-projected rays
        as a deterministic, shape-consistent placeholder.
        """
        b, t, v, j, _ = x2d.shape
        # Normalised image coordinates using intrinsics
        fx = K[:, :, 0, 0].unsqueeze(1).unsqueeze(3)  # (B, 1, V, 1)
        fy = K[:, :, 1, 1].unsqueeze(1).unsqueeze(3)
        cx = K[:, :, 0, 2].unsqueeze(1).unsqueeze(3)
        cy = K[:, :, 1, 2].unsqueeze(1).unsqueeze(3)

        x_norm = (x2d[..., 0] - cx) / fx  # (B, T, V, J)
        y_norm = (x2d[..., 1] - cy) / fy
        rays = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1)

        # Weighted average of rays as a coarse triangulation proxy
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        x3d = (weights[..., None] * rays).sum(dim=2)  # (B, T, J, 3)
        return x3d

    def forward(self, x2d, conf, K, cam):
        b, t, v, j, _ = x2d.shape

        # Build camera embedding per view/joint
        cam_embed = cam[:, None, :, None, :].expand(b, t, v, j, -1)
        features = torch.cat([x2d, conf[..., None], cam_embed], dim=-1)
        tokens = self.input_proj(features)  # (B, T, V, J, D)

        # Visibility head (per-view per-joint soft multiplier)
        visibility = self.vis_head(tokens).squeeze(-1)  # (B, T, V, J)

        # Factorised (T x V x J) attention
        refined = self._factorised_pass(tokens, visibility)

        # Uncertainty head -> log variance
        log_var = self.unc_head(refined).squeeze(-1)  # (B, T, V, J)
        weights = conf * visibility * torch.exp(-log_var)

        # Triangulation stub
        x3d_raw = self._triangulate_stub(x2d, weights, K)

        # Residual refinement: pool factorised features + geometry
        pooled = refined.mean(dim=(2, 3))  # (B, T, D)
        pooled = pooled[:, :, None, :].expand(-1, -1, j, -1)
        mlp_input = torch.cat([pooled, x3d_raw], dim=-1)
        x3d = x3d_raw + self.residual_mlp(mlp_input)

        return {
            "pose_3d": x3d,
            "visibility": visibility,
            "log_var": log_var,
        }


def main():
    torch.manual_seed(0)
    model = OmniMultiViewFusionSmoke(d_model=64, d_cam=16, n_layers=2, j=17)

    b, t, v, j = 2, 7, 4, 17
    x2d = torch.randn(b, t, v, j, 2)
    conf = torch.rand(b, t, v, j)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(b, v, -1, -1).clone()
    K[:, :, 0, 0] = 800.0
    K[:, :, 1, 1] = 800.0
    K[:, :, 0, 2] = 320.0
    K[:, :, 1, 2] = 240.0
    cam = torch.randn(b, v, 16)

    with torch.no_grad():
        out = model(x2d, conf, K, cam)

    assert out["pose_3d"].shape == (b, t, j, 3), out["pose_3d"].shape
    assert out["visibility"].shape == (b, t, v, j)
    assert out["log_var"].shape == (b, t, v, j)
    assert not torch.isnan(out["pose_3d"]).any()

    n_params = sum(p.numel() for p in model.parameters())
    print("OmniMultiViewFusionSmoke forward pass OK")
    print(f"  pose_3d shape     : {out['pose_3d'].shape}")
    print(f"  visibility shape  : {out['visibility'].shape}")
    print(f"  log_var shape     : {out['log_var'].shape}")
    print(f"  parameters        : {n_params:,}")


if __name__ == "__main__":
    main()
