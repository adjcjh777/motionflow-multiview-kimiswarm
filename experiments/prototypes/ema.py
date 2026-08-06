"""Exponential moving average (EMA) helpers for trainer checkpoints.

Drop-in utilities that add EMA parameter averaging and checkpoint resume
support without changing the default training path.  EMA is disabled when
``decay == 0``.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


class EMA:
    """Maintains an exponential moving average of a model's parameters/buffers.

    The shadow copy is updated in-place after each optimizer step.  For
    validation or checkpointing, ``apply_shadow`` temporarily swaps the model's
    parameters with the EMA shadow, and ``restore`` puts the online parameters
    back.

    Parameters
    ----------
    model: ``torch.nn.Module``
        Model whose parameters/buffers will be shadowed.
    decay: float
        EMA decay ``\beta`` (0.999 is typical).  Clamped to ``[0, 1)``.
    start_epoch: int
        1-indexed epoch at which to begin updating the shadow.  Updates before
        ``start_epoch`` are silently ignored.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, start_epoch: int = 1):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.start_epoch = max(1, start_epoch)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: torch.nn.Module) -> None:
        """Initialise the shadow from the current model state."""
        self.shadow.clear()
        for name, param in model.state_dict(keep_vars=True).items():
            if torch.is_floating_point(param):
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module, epoch: Optional[int] = None) -> None:
        """Advance the EMA shadow by one optimizer step.

        Parameters
        ----------
        model:
            The online model.
        epoch:
            Current 1-indexed epoch.  If provided and below ``start_epoch``,
            the update is skipped.
        """
        if epoch is not None and epoch < self.start_epoch:
            return
        one_minus_decay = 1.0 - self.decay
        for name, param in model.state_dict(keep_vars=True).items():
            if name in self.shadow:
                # shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].lerp_(param.detach().to(self.shadow[name].device), one_minus_decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Copy EMA shadow values into the model, backing up online values."""
        self._backup.clear()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            if name in self.shadow:
                self._backup[name] = buf.data.clone()
                buf.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """Restore online values saved by the last ``apply_shadow``."""
        if not self._backup:
            return
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        for name, buf in model.named_buffers():
            if name in self._backup:
                buf.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "start_epoch": self.start_epoch,
            "shadow": {k: v.cpu().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.start_epoch = int(state_dict["start_epoch"])
        self.shadow = {k: v.clone() for k, v in state_dict["shadow"].items()}


def save_checkpoint_with_ema(
    path: Any,
    model: torch.nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_val: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a resumable checkpoint, optionally containing an shadow EMA copy.

    The ``model`` argument is saved as the primary ``model`` key.  When
    ``ema`` is provided it is saved under the ``ema`` key so that resume can
    restore both the online weights and the shadow weights.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint_with_ema(
    path: Any,
    model: torch.nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
) -> Tuple[Optional[int], Optional[float]]:
    """Load a checkpoint produced by ``save_checkpoint_with_ema``.

    Plain state-dict checkpoints (no ``model`` key) are also supported for
    backward compatibility.

    Returns
    -------
    (epoch, best_val)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict.")

    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=strict)

    if ema is not None and "ema" in checkpoint and checkpoint["ema"] is not None:
        ema.load_state_dict(checkpoint["ema"])

    if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("epoch"), checkpoint.get("best_val")
