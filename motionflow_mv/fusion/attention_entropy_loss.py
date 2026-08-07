"""Attention-entropy regularisation for interpretable multi-view fusion.

The entropy loss encourages the per-view triangulation weight distribution to
concentrate mass on a small subset of views.  A one-hot weight vector has zero
entropy, while a uniform weight vector has the maximum entropy.  Adding the loss
to the training objective therefore pushes the model toward crisper, more
interpretable view selection.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionEntropyLoss(nn.Module):
    """Per-view triangulation-weight entropy regularisation.

    Parameters
    ----------
    weight:
        Scalar multiplier applied to the computed entropy.  ``0.0`` disables
        the loss (default).
    dim:
        Dimension along which the per-view weight distribution is defined.
        The default ``-2`` matches the view axis in the common v2/v3/v4
        weight tensors shapes ``(B, T, V, J)`` and ``(B, V, J)``.
    eps:
        Small constant for numerical stability.
    reduction:
        Reduction applied to the per-element entropy values.  ``'mean'``
        returns a scalar; ``'sum'`` returns the sum; ``'none'`` returns the
        unreduced tensor.

    Notes
    -----
    The implementation follows the formula in
    ``docs/v4_architecture_design_proposal.md`` Section 4.6:

        p = weights / (weights.sum(dim=view_dim) + eps)
        entropy = -sum(p * log(p + eps), dim=view_dim)

    where ``view_dim`` is the dimension that enumerates cameras.  The loss is
    non-negative, zero when the per-view weights are one-hot along ``dim``, and
    fully differentiable.
    """

    def __init__(
        self,
        weight: float = 0.0,
        dim: int = -2,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.weight = float(weight)
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute the entropy regularisation loss.

        Args
        ----
        weights:
            Non-negative per-view triangulation weights.  The view axis is
            determined by ``self.dim``.

        Returns
        -------
        Scalar loss (or unreduced tensor if ``reduction='none'``).
        """
        if torch.any(weights.isnan()):
            raise ValueError("AttentionEntropyLoss received NaN weights")
        if torch.any(weights < 0):
            raise ValueError("AttentionEntropyLoss expects non-negative weights")

        # Probability distribution over views.
        p = weights / (weights.sum(dim=self.dim, keepdim=True) + self.eps)
        # Shannon entropy along the view axis.
        entropy = -(p * torch.log(p + self.eps)).sum(dim=self.dim)

        if self.reduction == "mean":
            entropy = entropy.mean()
        elif self.reduction == "sum":
            entropy = entropy.sum()

        return self.weight * entropy

    def extra_repr(self) -> str:
        return f"weight={self.weight}, dim={self.dim}, reduction={self.reduction!r}"


if __name__ == "__main__":
    B, T, V, J = 2, 5, 4, 17

    # Uniform weights -> maximum entropy (non-negative).
    uniform = torch.ones(B, T, V, J, dtype=torch.float32)
    loss_fn = AttentionEntropyLoss(weight=0.01, dim=-2)
    loss_uniform = loss_fn(uniform)
    assert loss_uniform.numel() == 1
    assert loss_uniform.item() > 0.0, "Uniform weights must give positive entropy"

    # One-hot weights -> zero entropy.
    one_hot = torch.zeros(B, T, V, J, dtype=torch.float32)
    one_hot[:, :, 0, :] = 1.0
    loss_one_hot = loss_fn(one_hot)
    assert abs(loss_one_hot.item()) < 1e-5, "One-hot weights must give zero entropy"

    # Differentiability.
    weights = torch.rand(B, T, V, J, requires_grad=True)
    loss = loss_fn(weights)
    loss.backward()
    assert weights.grad is not None
    assert weights.grad.shape == weights.shape

    print(f"AttentionEntropyLoss CPU smoke test passed (uniform entropy loss={loss_uniform.item():.6f})")
