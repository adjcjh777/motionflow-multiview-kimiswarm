"""Differentiable confidence-weighted DLT triangulation.

Given per-view 2D points and learned weights, solve for 3D via DLT in a
differentiable way so it can be trained end-to-end with reprojection loss.
"""

import torch
import torch.nn as nn


class RobustTriangulationModel(nn.Module):
    """Predict per-view weights and triangulate.

    Input:  (B, V, J, 3) -> (x, y, confidence)
    Output: (B, J, 3) 3D positions
    """

    def __init__(self, j: int = 17, d: int = 32, n_views: int = 4):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.lift = nn.Linear(3, d)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
        self.weight_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, proj_matrices: torch.Tensor) -> torch.Tensor:
        # x: (B, V, J, 3), proj_matrices: (V, 3, 4)
        B, V, J, _ = x.shape
        x_emb = self.lift(x)  # (B, V, J, D)
        x_emb = x_emb.permute(0, 2, 1, 3).reshape(B * J, V, self.d)
        attn_out, _ = self.attn(x_emb, x_emb, x_emb)  # (B*J, V, D)
        weights = torch.sigmoid(self.weight_head(attn_out)).squeeze(-1)  # (B*J, V)
        weights = weights.view(B, J, V)  # (B, J, V)

        # Differentiable weighted DLT for each joint
        # proj_matrices: (V, 3, 4)
        points_2d = x[..., :2]  # (B, V, J, 2)
        pred_3d = []
        for j_idx in range(J):
            p2d = points_2d[:, :, j_idx, :]  # (B, V, 2)
            w = weights[:, j_idx, :]  # (B, V)
            X = self._triangulate(p2d, w, proj_matrices)  # (B, 3)
            pred_3d.append(X)
        pred_3d = torch.stack(pred_3d, dim=1)  # (B, J, 3)
        return pred_3d

    def _triangulate(self, points_2d: torch.Tensor, weights: torch.Tensor, proj_matrices: torch.Tensor):
        """Differentiable weighted DLT for a single joint using a stable
        inhomogeneous pseudo-inverse.

        points_2d: (B, V, 2)
        weights: (B, V)
        proj_matrices: (V, 3, 4)
        """
        B, V, _ = points_2d.shape
        # Build A (B, 2V, 4) for the DLT equations.
        A = []
        for v in range(V):
            P = proj_matrices[v]  # (3, 4)
            x = points_2d[:, v, 0]  # (B,)
            y = points_2d[:, v, 1]  # (B,)
            A.append(x[:, None] * P[2:3, :] - P[0:1, :])  # (B, 4)
            A.append(y[:, None] * P[2:3, :] - P[1:2, :])
        A = torch.stack(A, dim=1)  # (B, 2V, 4)

        # Inhomogeneous weighted least squares: fix last coordinate to 1 and
        # solve A[:, :, :3] X' = -A[:, :, 3] with weights.
        A3 = A[:, :, :3]  # (B, 2V, 3)
        a4 = A[:, :, 3:]  # (B, 2V, 1)
        w = weights.unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * V, 1)  # (B, 2V, 1)
        Aw = A3 * torch.sqrt(w + 1e-6)
        bw = -a4 * torch.sqrt(w + 1e-6)
        # lstsq is differentiable and more stable than SVD for small systems.
        X, *_ = torch.linalg.lstsq(Aw, bw)
        X = X.squeeze(-1)  # (B, 3)
        return X
