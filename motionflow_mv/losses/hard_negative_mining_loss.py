"""Failure-driven hard-negative mining loss for multi-view 3D pose.

This module implements online hard-example mining (OHEM) for per-joint 3D
estimation.  During training it up-weights the highest-error joints so the
model spends more capacity on its own failure modes.  A small FIFO memory bank
keeps the hardest samples of the current epoch in order to combat catastrophic
forgetting of rare cases.
"""

from collections import deque
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeMiningLoss(nn.Module):
    """Per-joint hard-negative mining loss.

    Parameters
    ----------
    base_loss:
        Base loss type.  ``'mse'`` or ``'l1'``.
    ohem_ratio:
        Fraction of joints per batch selected as hard negatives.
    hard_weight:
        Multiplicative weight applied to hard-negative joints.
    memory_size:
        Size of the FIFO hard-sample memory bank.  ``0`` disables the bank.
    memory_prob:
        Probability (0..1) of replacing a kept sample in the memory bank with a
        current hard negative.  ``0`` keeps the bank read-only after warm-up.
    """

    def __init__(
        self,
        base_loss: str = "mse",
        ohem_ratio: float = 0.25,
        hard_weight: float = 2.0,
        memory_size: int = 256,
        memory_prob: float = 0.0,
    ):
        super().__init__()
        if base_loss not in ("mse", "l1"):
            raise ValueError("base_loss must be 'mse' or 'l1'")
        self.base_loss = base_loss
        self.ohem_ratio = ohem_ratio
        self.hard_weight = hard_weight
        self.memory_size = memory_size
        self.memory_prob = memory_prob
        self._memory: deque = deque(maxlen=memory_size)

    def _compute_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return per-joint L2 distance ``(B, ..., J)``.

        Args:
            pred:   (B, ..., J, 3)
            target: same shape as pred
        Returns:
            error:  (B, ..., J)
        """
        return (pred - target).norm(dim=-1)

    def _base_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Element-wise base loss with optional confidence/validity weights.

        Args:
            pred:    (B, ..., J, 3)
            target:  same shape as pred
            weights: optional (B, ..., J) positive weights
        Returns:
            loss:    (B, ..., J) element-wise loss
        """
        if self.base_loss == "mse":
            loss = ((pred - target) ** 2).sum(dim=-1)
        else:  # l1
            loss = (pred - target).norm(dim=-1)

        if weights is not None:
            loss = loss * weights
        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute hard-negative reweighted loss.

        Args:
            pred:    (B, ..., J, 3) predicted 3D joints.
            target:  same shape as pred.
            weights: optional (B, ..., J) per-joint weight mask.
            return_mask:
                If ``True``, also return the binary hard-negative mask.

        Returns:
            Scalar loss.  If ``return_mask`` is ``True``, also returns the
            hard-negative mask with the same leading shape as the per-joint
            loss.
        """
        base = self._base_loss(pred, target, weights)  # (B, ..., J)
        with torch.no_grad():
            error = self._compute_error(pred, target)  # (B, ..., J)
            flat_error = error.view(-1)
            k = max(1, int(flat_error.numel() * self.ohem_ratio))
            topk_threshold = torch.topk(flat_error, k, largest=True).values.min()
            hard_mask = error >= topk_threshold  # (B, ..., J)
            if weights is not None:
                hard_mask = hard_mask & (weights > 0)

        reweighted = base.clone()
        reweighted[hard_mask] = reweighted[hard_mask] * self.hard_weight
        if weights is not None:
            denom = weights.sum() + 1e-8
        else:
            denom = base.numel()
        loss = reweighted.sum() / denom

        # Memory bank update (no gradient).
        if self.training and self.memory_size > 0:
            self._update_memory(pred.detach(), target.detach(), error.detach())

        if return_mask:
            return loss, hard_mask
        return loss

    def _update_memory(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        error: torch.Tensor,
    ) -> None:
        """Push the single hardest sample of the current batch into the FIFO bank."""
        flat_error = error.view(error.size(0), -1).mean(dim=1)  # per-sample error
        hardest_idx = int(flat_error.argmax())
        sample_key = (pred[hardest_idx], target[hardest_idx])
        if len(self._memory) < self.memory_size:
            self._memory.append(sample_key)
        elif self.memory_prob > 0 and torch.rand(1).item() < self.memory_prob:
            self._memory.append(sample_key)

    def sample_memory(
        self,
        n: int,
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a stacked batch of ``n`` hard samples from the memory bank.

        Args:
            n: Number of samples to draw.  If the bank is smaller, all are returned.
            device: Target device for the returned tensors.

        Returns:
            Tuple ``(pred, target)`` or ``None`` if the bank is empty.
        """
        if not self._memory:
            return None
        import random

        n = min(n, len(self._memory))
        sampled = random.sample(self._memory, n)
        preds = torch.stack([s[0] for s in sampled], dim=0)
        targets = torch.stack([s[1] for s in sampled], dim=0)
        if device is not None:
            preds = preds.to(device)
            targets = targets.to(device)
        return preds, targets

    def state_dict(self) -> dict:
        """Return serialisable state (memory bank is not saved to keep checkpoints small)."""
        return {
            "ohem_ratio": self.ohem_ratio,
            "hard_weight": self.hard_weight,
            "memory_size": self.memory_size,
            "memory_prob": self.memory_prob,
        }


def focal_hard_negative_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal-style reweighting of the MSE loss by prediction error.

    A softer alternative to the top-k OHEM loss above; down-weights easy joints
    and keeps gradient on all examples.

    Args:
        pred:    (B, ..., J, 3)
        target:  same shape as pred
        gamma:   focal exponent
        weights: optional (B, ..., J) per-joint weight mask
    Returns:
        Scalar loss
    """
    error = (pred - target).norm(dim=-1)  # (B, ..., J)
    max_error = error.detach().max() + 1e-8
    focal_weight = (error / max_error) ** gamma
    loss = focal_weight * error
    if weights is not None:
        loss = loss * weights
    return loss.mean()
