"""Minimal demonstration of camera positional encoding for variable-view fusion.

The script builds two camera rigs (4 views and 14 views), shows that the
encoding is scale-invariant, and demonstrates how to replace the learned
view_pos_embed in a temporal/cross-view ray-attention model.
"""

import math
import torch
import torch.nn as nn


def fourier_features(x, L=6):
    """NeRF-style positional encoding for a scalar tensor x (any shape).

    Args:
        x: tensor of scalars, shape (...).
        L: number of frequency bands.

    Returns:
        Encoded tensor of shape (..., 2*L + 1) in [-1, 1].
    """
    freq = 2.0 ** torch.linspace(0, L - 1, L, device=x.device, dtype=x.dtype)
    x_expanded = x.unsqueeze(-1) * math.pi * freq  # (..., L)
    return torch.cat([x.unsqueeze(-1), torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)


class CameraPositionalEncoding(nn.Module):
    """Geometry-based camera positional encoding.

    Input: K (B, V, 3, 3), R (B, V, 3, 3), t (B, V, 3)
    Output: (B, V, d)
    """

    def __init__(self, d=64, L=6):
        super().__init__()
        self.d = d
        self.L = L
        # Centers (3) + principal ray (3) + normalized focal length (1)
        # each mapped to (2*L + 1) scalars.
        in_dim = (3 + 3 + 1) * (2 * L + 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, K, R, t):
        eps = 1e-6
        # Camera center in world: c = -R^T t
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (B, V, 3)
        centered = centers - centers.mean(dim=1, keepdim=True)
        scale = centered.norm(dim=-1).max(dim=1, keepdim=True)[0].clamp_min(eps)
        centers_norm = centered / scale.unsqueeze(-1)

        # Principal ray in world: r = R^T [0,0,1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=R.device, dtype=R.dtype)
        principal = torch.einsum("bvij,j->bvi", R.transpose(-2, -1), ez)

        # Mean focal length, normalized by mean across views.
        f = ((K[..., 0, 0] + K[..., 1, 1]) * 0.5)
        f_norm = f / (f.mean(dim=1, keepdim=True) + eps)

        components = []
        for vec in [centers_norm, principal, f_norm.unsqueeze(-1)]:
            enc = fourier_features(vec, self.L)
            components.append(enc.reshape(enc.shape[0], enc.shape[1], -1))
        camera_feat = torch.cat(components, dim=-1)

        return self.mlp(camera_feat)


def build_circular_rig(n_views, radius=3.0, height=1.0):
    """Return (K, R, t) for a simple circular rig looking at the origin."""
    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = torch.tensor([radius * math.cos(theta), radius * math.sin(theta), height])
        forward = -c / c.norm()
        up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.cross(forward, up, dim=-1)
        right /= right.norm()
        up = torch.cross(right, forward, dim=-1)
        R = torch.stack([right, up, -forward], dim=0)
        t = -R @ c
        K = torch.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    K = torch.stack(Ks).unsqueeze(0)  # (1, V, 3, 3)
    R = torch.stack(Rs).unsqueeze(0)
    t = torch.stack(ts).unsqueeze(0)
    return K, R, t


def build_scaled_rig_pair(n_views=4):
    """Return two rigs with identical orientation but different scale."""
    # Fixed orientation: looking roughly along +x, with world up.
    base_R = torch.tensor([[0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0],
                           [1.0, 0.0, 0.0]])
    Ks, Rs, ts1, ts2 = [], [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c1 = torch.tensor([3.0 * math.cos(theta), 3.0 * math.sin(theta), 1.0])
        c2 = 2.0 * c1
        t1 = -base_R @ c1
        t2 = -base_R @ c2
        K = torch.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        Ks.append(K)
        Rs.append(base_R)
        ts1.append(t1)
        ts2.append(t2)
    K = torch.stack(Ks).unsqueeze(0)
    R = torch.stack(Rs).unsqueeze(0)
    t1 = torch.stack(ts1).unsqueeze(0)
    t2 = torch.stack(ts2).unsqueeze(0)
    return K, R, t1, t2


def demo_variable_view():
    enc = CameraPositionalEncoding(d=64)
    for n_views in [4, 14]:
        K, R, t = build_circular_rig(n_views)
        emb = enc(K, R, t)
        print(f"n_views={n_views:2d} -> embedding shape {emb.shape}")


def demo_scale_invariance():
    enc = CameraPositionalEncoding(d=64)
    K, R, t1, t2 = build_scaled_rig_pair(4)
    emb1 = enc(K, R, t1)
    emb2 = enc(K, R, t2)
    diff = (emb1 - emb2).abs().max().item()
    print(f"scale-invariance max diff: {diff:.6f} (should be ~0)")


def demo_model_patch():
    """Stub showing the injection point in a temporal/cross-view model."""
    B, T, V, J, d = 2, 13, 4, 17, 64
    feat = torch.randn(B, T, V, J, d)
    K, R, t = build_circular_rig(V)

    # camera-derived view embedding replaces learned nn.Parameter(n_views, d)
    enc = CameraPositionalEncoding(d=d)
    view_emb = enc(K, R, t)  # (1, V, d)
    view_emb = view_emb.view(1, 1, V, 1, d)  # broadcast over B, T, J
    feat = feat + view_emb
    print(f"patched feature shape: {feat.shape}")


if __name__ == "__main__":
    demo_variable_view()
    demo_scale_invariance()
    demo_model_patch()
