"""Cross-view contrastive loss for multi-view 3D pose representations.

Encourages the per-joint features produced by each view to be *view-invariant*
by pulling together embeddings of the **same joint across different views** and
pushing apart embeddings of **different joints** (intra- and cross-view).  The
loss is purely auxiliary and does not change the differentiable triangulation
path; it can be dropped in by adding one extra term to the training objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewJointContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss on per-joint multi-view features.

    Parameters
    ----------
    d:
        Dimensionality of the input feature tensor.
    projection_dim:
        Dimensionality of the projected embedding used for contrastive learning.
    temperature:
        Softmax temperature for the contrastive objective.
    """

    def __init__(
        self,
        d: int,
        projection_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self.temperature = temperature

    def forward(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross-view joint contrastive loss.

        Args:
            feat: Per-joint multi-view features of shape ``(N, V, J, d)``,
                where ``N`` is the batch (clip) dimension, ``V`` the number of
                views, ``J`` the number of joints and ``d`` the feature dim.
            mask: Optional per-anchor validity mask of shape ``(N, V, J)``.
                If provided, invalid anchors are excluded from the mean.

        Returns:
            Scalar loss tensor.
        """
        if feat.dim() != 4:
            raise ValueError(f"Expected feat to be 4-D (N,V,J,d), got {feat.shape}")

        N, V, J, _ = feat.shape
        if V < 2:
            return torch.tensor(0.0, device=feat.device, dtype=feat.dtype)

        # Project and L2-normalise embeddings.
        z = F.normalize(self.projection(feat), dim=-1)  # (N, V, J, C)

        # Pairwise cosine similarities: sim[n, v, j, vp, jp] = z[n,v,j]·z[n,vp,jp]
        sim = torch.einsum("nvjc,nwkc->nvjwk", z, z) / self.temperature
        # sim: (N, V, J, V, J)

        # Build anchor/candidate index masks of shape (V, J, V, J).
        v_a = torch.arange(V, device=feat.device).view(V, 1, 1, 1)
        j_a = torch.arange(J, device=feat.device).view(1, J, 1, 1)
        v_c = torch.arange(V, device=feat.device).view(1, 1, V, 1)
        j_c = torch.arange(J, device=feat.device).view(1, 1, 1, J)

        self_mask = (v_a == v_c) & (j_a == j_c)  # exact same token
        pos_mask = (j_a == j_c) & ~self_mask    # same joint, different view
        neg_mask = j_a != j_c                     # different joint (any view)

        A = V * J
        sim_flat = sim.reshape(N, A, A)
        self_flat = self_mask.reshape(A, A)
        pos_flat = pos_mask.reshape(A, A)

        # Numerator: log-sum-exp over positive (same-joint cross-view) candidates.
        pos_sim = sim_flat.masked_fill(~pos_flat[None, :, :], -1e9)
        numerator = torch.logsumexp(pos_sim, dim=-1)  # (N, A)

        # Denominator: log-sum-exp over all non-self candidates.
        denom_sim = sim_flat.masked_fill(self_flat[None, :, :], -1e9)
        denominator = torch.logsumexp(denom_sim, dim=-1)  # (N, A)

        loss = -(numerator - denominator)  # (N, A)

        if mask is not None:
            mask_flat = mask.reshape(N, A).float()
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss
