"""Regularisation losses for the dynamic view-selection gate.

The gate is trained with the pose loss plus a small sparsity-entropy
regulariser that encourages it to drop noisy/occluded views without
collapsing to all-zeros or all-ones.
"""

import torch
import torch.nn as nn


class ViewSelectionLoss(nn.Module):
    """Sparsity + entropy regulariser for soft view-selection gates.

    Parameters
    ----------
    sparsity_weight:
        Weight of the mean gate penalty (higher = larger average gate).
    entropy_weight:
        Weight of the binary-entropy penalty (higher = pushes gates toward 0/1).
    """

    def __init__(self, sparsity_weight: float = 0.01, entropy_weight: float = 0.001):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight

    def forward(self, gate_weights: torch.Tensor):
        """Return the total regularisation loss.

        Args
        ----
        gate_weights:
            Soft gate tensor of any shape in ``[0, 1]``.

        Returns
        -------
        Scalar regularisation loss.
        """
        sparsity = gate_weights.mean()
        entropy = -(gate_weights * torch.log(gate_weights + 1e-6) +
                    (1 - gate_weights) * torch.log(1 - gate_weights + 1e-6)).mean()
        return self.sparsity_weight * sparsity + self.entropy_weight * entropy
