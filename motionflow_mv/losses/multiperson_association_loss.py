"""Association loss for multi-person multi-view 3-D pose estimation.

Provides a per-joint 3-D MSE that is summed over all people plus an optional
distinctiveness term that keeps different people from collapsing to the same
3-D location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPersonAssociationLoss(nn.Module):
    """Multi-person 3-D pose + person-distinctiveness loss.

    Parameters
    ----------
    distinctiveness_weight:
        Weight for the pairwise centre-repulsion term.  ``0.0`` disables it.
    min_center_distance:
        Target minimum distance (in metres) between any two people.  Distances
        below this value are penalized.
    """

    def __init__(
        self,
        distinctiveness_weight: float = 0.01,
        min_center_distance: float = 1.0,
    ):
        super().__init__()
        self.distinctiveness_weight = distinctiveness_weight
        self.min_center_distance = min_center_distance

    def forward(
        self,
        pred_3d: torch.Tensor,
        gt_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-person loss.

        Parameters
        ----------
        pred_3d:
            Predicted 3-D joints, shape ``(B, T, P, J, 3)``.
        gt_3d:
            Ground-truth 3-D joints, shape ``(B, T, P, J, 3)``.

        Returns
        -------
        Scalar loss tensor.
        """
        mse = F.mse_loss(pred_3d, gt_3d)

        if self.distinctiveness_weight == 0.0 or pred_3d.shape[2] < 2:
            return mse

        # Person centres are the mean of all joints for each person.
        centres = pred_3d.mean(dim=-2)  # (B, T, P, 3)
        # Pairwise differences: (B, T, P, P, 3)
        diffs = centres.unsqueeze(3) - centres.unsqueeze(2)
        dists = diffs.norm(dim=-1)  # (B, T, P, P)
        # Mask out diagonal (same person).
        mask = torch.ones_like(dists, dtype=torch.bool)
        mask.diagonal(dim1=-2, dim2=-1).fill_(False)
        distinct = torch.relu(self.min_center_distance - dists[mask]).mean()
        return mse + self.distinctiveness_weight * distinct
