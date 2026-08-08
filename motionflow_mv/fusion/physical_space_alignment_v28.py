"""v28: Physical-space alignment for multi-view 3D human pose."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

class PhysicalSpaceAlignmentV28(nn.Module):
    def __init__(self, j, hidden=64, max_residual=0.05, dropout=0.1):
        super().__init__()
        self.j = j
        self.max_residual = max_residual
        self.gravity_dir = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]), requires_grad=False)
        self.refiner = nn.Sequential(
            nn.Linear(6, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3), nn.Tanh(),
        )
        for p in self.refiner[-2].parameters():
            nn.init.zeros_(p)
        self.residual_logit = nn.Parameter(torch.tensor(-6.0))

    @property
    def residual_scale(self):
        return torch.sigmoid(self.residual_logit)

    def forward(self, X, gravity_dir=None, return_reg_loss=False):
        if gravity_dir is None:
            gravity_dir = self.gravity_dir
        B, T, J, _ = X.shape
        g = gravity_dir.to(X.device, X.dtype).view(1, 1, 1, 3).expand(B, T, J, -1)
        feat = torch.cat([X, g], dim=-1)
        raw = self.refiner(feat)
        residual = self.max_residual * raw
        scale = self.residual_scale
        out = X + scale * residual
        if not return_reg_loss:
            return out
        applied = scale * residual
        return out, applied.pow(2).mean()


def floor_loss(X, floor_height, foot_joint_indices, gravity_dir=None, floor_quantile=0.05):
    if gravity_dir is None:
        gravity_dir = torch.tensor([0.0, 1.0, 0.0], device=X.device, dtype=X.dtype)
    g = gravity_dir / (gravity_dir.norm() + 1e-8)
    h = torch.einsum("btjc,c->btj", X, g)
    feet = h[:, :, foot_joint_indices]
    n = feet.shape[-1]
    if n > 1:
        k = max(1, int(floor_quantile * n))
        floor_h, _ = torch.topk(feet, k, dim=-1, largest=False)
        floor_h = floor_h[..., -1]
    else:
        floor_h = feet[..., 0]
    return (floor_h.unsqueeze(-1) - feet).clamp(min=0.0).mean()


def bone_temporal_loss(X, parents):
    if X.shape[1] < 2:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
    bone_vecs = [X[..., c, :] - X[..., p, :] for c, p in enumerate(parents) if p >= 0]
    if not bone_vecs:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
    bones = torch.stack(bone_vecs, dim=-2)
    lengths = bones.norm(dim=-1)
    return (lengths[:, 1:] - lengths[:, :-1]).pow(2).mean()
