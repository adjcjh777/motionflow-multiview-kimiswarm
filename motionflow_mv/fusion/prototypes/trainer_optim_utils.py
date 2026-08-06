"""Trainer optim utilities: warmup + cosine LR, gradient clipping, AMP.

This module is a prototype for the iter-next-swarm "Trainer" task.  It is
intended to be imported by the MPI-INF-3DHP trainer and provides small,
self-contained helpers for:

* cosine learning-rate schedule with optional linear warmup
* gradient clipping
* automatic mixed precision (AMP) with a CPU-safe no-op fallback
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn, optim


# ---------------------------------------------------------------------------
# Warmup + Cosine LR schedule
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 0,
    eta_min: float = 0.0,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Return a scheduler that linearly warms up then cosine anneals LR.

    The warmup phase runs for ``warmup_epochs`` epochs with the learning rate
    increasing linearly from 0 to the initial lr set in ``optimizer``.  After
    warmup, the LR follows a cosine annealing curve down to ``eta_min`` over
    the remaining epochs.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        total_epochs: Total number of epochs for the run.
        warmup_epochs: Number of epochs used for linear warmup (>= 0).
        eta_min: Minimum learning rate at the end of cosine annealing.

    Returns:
        A scheduler, or ``None`` if ``total_epochs <= 0``.
    """
    if total_epochs <= 0:
        return None
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    return _WarmupCosineLR(optimizer, total_epochs, warmup_epochs, eta_min)


class _WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    """Linear warmup + cosine decay LR scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.base_lrs_initial = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup from 0 to base_lr.
            alpha = (epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs_initial]
        # Cosine annealing.
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs_initial
        ]


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_gradients(model: nn.Module, max_norm: Optional[float] = None) -> Optional[float]:
    """Clip gradients of all model parameters by global L2 norm.

    Args:
        model: Model whose parameters will be clipped.
        max_norm: Maximum allowable gradient norm.  If ``None`` or ``<= 0``,
            no clipping is performed.

    Returns:
        The total gradient norm before clipping (useful for logging), or
        ``None`` if clipping was skipped.
    """
    if max_norm is None or max_norm <= 0.0:
        return None
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


# ---------------------------------------------------------------------------
# AMP (Automatic Mixed Precision)
# ---------------------------------------------------------------------------

class AMPContext:
    """CPU-safe AMP context manager + gradient scaler.

    On CUDA devices, this wraps ``torch.cuda.amp.GradScaler`` and
    ``torch.cuda.amp.autocast``.  On CPU, autocast is a no-op and the scaler
    is disabled (gradients are not actually scaled), which still allows the
    same training code to run for smoke tests.

    Args:
        enabled: Whether AMP is enabled at all.  When ``False``, ``autocast``
            and ``scale`` become no-ops.
        device: Target device; used to decide whether CUDA AMP can be used.
    """

    def __init__(self, enabled: bool = True, device: Optional[torch.device] = None):
        self.enabled = enabled and (device is None or device.type == "cuda")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)
        self._autocast = torch.amp.autocast("cuda", enabled=self.enabled)

    def __enter__(self):
        self._autocast.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._autocast.__exit__(exc_type, exc_val, exc_tb)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale a loss value for gradient scaling."""
        if self.enabled:
            return self._scaler.scale(loss)
        return loss

    def unscale(self, optimizer: optim.Optimizer) -> None:
        """Unscale gradients prior to clipping/inspection."""
        if self.enabled:
            self._scaler.unscale_(optimizer)

    def step(self, optimizer: optim.Optimizer) -> None:
        """Perform an optimizer step, unscaling gradients first if needed."""
        if self.enabled:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update the gradient scaler after an optimization step."""
        if self.enabled:
            self._scaler.update()

    def state_dict(self) -> dict:
        return self._scaler.state_dict() if self.enabled else {}

    def load_state_dict(self, state_dict: dict) -> None:
        if self.enabled:
            self._scaler.load_state_dict(state_dict)
