"""Trainer v2 for MotionFlow-MultiView.

Implements a reusable training loop with the following improvements over the
legacy ad-hoc training scripts:

* cosine learning-rate schedule with optional linear warmup
* global L2 gradient clipping
* automatic mixed precision (AMP) with a CPU-safe fallback
* exponential moving average (EMA) of model parameters
* generic ``TrainerV2`` plus a convenience ``MultiViewPoseTrainerV2`` for the
  common ``(x, y, K, R, t)`` multi-view pose batch convention

The design is intentionally isolated in ``motionflow_mv/training/`` so that it
can be adopted incrementally without modifying existing shared modules.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast as amp_autocast


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 0,
    eta_min: float = 0.0,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Return a linear-warmup + cosine-annealing LR scheduler.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        total_epochs: Total number of epochs for the run.
        warmup_epochs: Number of epochs used for linear warmup (>= 0).
        eta_min: Minimum learning rate at the end of cosine annealing.

    Returns:
        A scheduler, or ``None`` if ``total_epochs <= 0``.
    """
    if total_epochs <= 0:
        return None
    return _WarmupCosineLR(optimizer, total_epochs, max(0, min(warmup_epochs, total_epochs)), eta_min)


class _WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay (epoch-level)."""

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

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = (epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs_initial]
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs_initial
        ]


# ---------------------------------------------------------------------------
# AMP context (CPU-safe)
# ---------------------------------------------------------------------------

class _AMPContext:
    """CPU-safe AMP helper wrapping ``autocast`` and ``GradScaler``.

    CUDA AMP is only enabled when the target device is CUDA *and* the user
    requested AMP. On CPU, ``autocast`` is a no-op and ``GradScaler`` is kept
    disabled, so the same training code can run on CPU-only smoke tests.
    """

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled and device.type == "cuda"
        if self.enabled:
            from torch.amp import GradScaler
            self._scaler: Optional[GradScaler] = GradScaler("cuda", enabled=True)  # type: ignore[arg-type]
            self._autocast = amp_autocast("cuda", enabled=True)
        else:
            self._scaler = None
            self._autocast = amp_autocast("cuda", enabled=False)

    def __enter__(self) -> "_AMPContext":
        self._autocast.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
        return self._autocast.__exit__(exc_type, exc_val, exc_tb)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.enabled and self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def unscale(self, optimizer: optim.Optimizer) -> None:
        if self.enabled and self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def step(self, optimizer: optim.Optimizer) -> None:
        if self.enabled and self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        if self.enabled and self._scaler is not None:
            self._scaler.update()

    def state_dict(self) -> Dict[str, Any]:
        if self.enabled and self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.enabled and self._scaler is not None:
            self._scaler.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Exponential moving average
# ---------------------------------------------------------------------------

class EMA:
    """Maintain an exponential moving average of a model's parameters.

    The shadow parameters can be swapped into the model for evaluation and
    restored afterwards, allowing the original training weights to remain
    untouched.

    Args:
        model: Model to shadow.
        decay: EMA decay coefficient (typically 0.999 - 0.9999).  Use
            ``update_every`` for bias correction if starting from scratch.
        update_every: Apply EMA update once every N optimizer steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, update_every: int = 1):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        self.decay = decay
        self.update_every = update_every
        self._step = 0
        self.shadow: Dict[str, torch.Tensor] = OrderedDict()
        self._backup: Optional[Dict[str, torch.Tensor]] = None
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update the EMA shadow with the current model parameters."""
        self._step += 1
        if self._step % self.update_every != 0:
            return
        decay = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            shadow = self.shadow[name]
            if shadow.dtype.is_floating_point:
                self.shadow[name] = shadow * decay + param.data * (1.0 - decay)
            else:
                # Integer / bool tensors are not expected for float params;
                # keep the latest value to be safe.
                self.shadow[name] = param.data.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        """Copy EMA shadow into the model and keep a backup of current params."""
        if self._backup is not None:
            warnings.warn("apply_shadow called twice without restore(); restoring first", stacklevel=2)
            self.restore(model)
        self._backup = OrderedDict()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self._backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore the model parameters saved by the last ``apply_shadow``."""
        if self._backup is None:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._backup:
                    param.data.copy_(self._backup[name])
        self._backup = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "step": self._step,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.update_every = state_dict["update_every"]
        self._step = state_dict["step"]
        self.shadow = OrderedDict({k: v.clone() for k, v in state_dict["shadow"].items()})


# ---------------------------------------------------------------------------
# Generic trainer v2
# ---------------------------------------------------------------------------

class TrainerV2:
    """Generic trainer with cosine LR, warmup, gradient clipping, AMP, and EMA.

    The trainer is model-agnostic: it receives batches from a DataLoader and a
    user-provided ``compute_loss`` function that returns a scalar loss and an
    metrics dictionary.

    Args:
        model: The model to train.
        optimizer: Optimizer for the model parameters.
        device: Device to run on.
        compute_loss: Callable ``(model, batch, device) -> (loss, metrics)``.
            If ``None``, the subclass must override ``compute_loss``.
        total_epochs: Total number of training epochs (used to build the
            cosine+warmup scheduler). Ignored if ``scheduler`` is provided.
        warmup_epochs: Number of linear-warmup epochs.
        eta_min: Minimum LR at the end of cosine annealing.
        scheduler: Optional external LR scheduler. If provided, overrides
            the internally built warmup+cosine scheduler.
        max_grad_norm: Maximum global L2 norm for gradient clipping.  No
            clipping if ``None`` or ``<= 0``.
        amp_enabled: Whether to enable automatic mixed precision.
        ema_decay: EMA decay.  Set to ``None`` to disable EMA.
        ema_update_every: Update EMA once every N optimizer steps.
        ema_eval: If ``True``, evaluation uses the EMA shadow weights.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        *,
        compute_loss: Optional[Callable[[nn.Module, Any, torch.device], Tuple[torch.Tensor, Dict[str, Any]]]] = None,
        total_epochs: int = 0,
        warmup_epochs: int = 0,
        eta_min: float = 0.0,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: Optional[float] = None,
        amp_enabled: bool = True,
        ema_decay: Optional[float] = 0.999,
        ema_update_every: int = 1,
        ema_eval: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        if compute_loss is not None:
            self.compute_loss = compute_loss  # type: ignore[assignment]
        if scheduler is not None:
            self.scheduler = scheduler
        elif total_epochs > 0:
            self.scheduler = build_lr_scheduler(optimizer, total_epochs, warmup_epochs, eta_min)
        else:
            self.scheduler = None
        self.max_grad_norm = max_grad_norm
        self.amp = _AMPContext(enabled=amp_enabled, device=device)
        self.ema: Optional[EMA] = None
        if ema_decay is not None:
            self.ema = EMA(model, decay=ema_decay, update_every=ema_update_every)
        self.ema_eval = ema_eval
        self.epoch = 0
        self.history: List[Dict[str, Any]] = []

    def compute_loss(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss for a training batch.

        Subclasses or users can override this. The default raises an error.
        """
        raise NotImplementedError("compute_loss must be provided or overridden")

    def _clip_gradients(self) -> Optional[float]:
        if self.max_grad_norm is None or self.max_grad_norm <= 0.0:
            return None
        return nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()

    def train_step(self, batch: Any) -> Dict[str, Any]:
        """Run a single training step and return metrics."""
        self.model.train()
        self.optimizer.zero_grad()

        with self.amp:
            loss, metrics = self.compute_loss(self.model, batch, self.device)
            scaled_loss = self.amp.scale(loss)

        scaled_loss.backward()
        self.amp.unscale(self.optimizer)
        grad_norm = self._clip_gradients()
        self.amp.step(self.optimizer)
        self.amp.update()

        if self.ema is not None:
            self.ema.update(self.model)

        step_metrics = {"loss": loss.detach().item()}
        step_metrics.update({k: v.detach().item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()})
        if grad_norm is not None:
            step_metrics["grad_norm"] = grad_norm
        return step_metrics

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch and return averaged metrics."""
        self.model.train()
        running: Dict[str, float] = {}
        count = 0
        for batch in dataloader:
            step_metrics = self.train_step(batch)
            count += 1
            for k, v in step_metrics.items():
                running[k] = running.get(k, 0.0) + v
        return {k: v / count for k, v in running.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_metric: Optional[Callable[[nn.Module, Any, torch.device], Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Evaluate the model and return metrics.

        If EMA is enabled and ``ema_eval`` is ``True``, the EMA shadow weights
        are applied during evaluation and restored afterwards.
        """
        self.model.eval()
        apply_ema = self.ema is not None and self.ema_eval
        if apply_ema:
            self.ema.apply_shadow(self.model)  # type: ignore[union-attr]

        try:
            running: Dict[str, float] = {}
            count = 0
            for batch in dataloader:
                if compute_metric is not None:
                    metrics = compute_metric(self.model, batch, self.device)
                else:
                    # Default evaluation assumes the model returns the first
                    # output element of a tuple and the batch has a target.
                    metrics = self._default_eval_metric(batch)
                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
                count += 1
            result = {k: v / count for k, v in running.items()}
        finally:
            if apply_ema:
                self.ema.restore(self.model)  # type: ignore[union-attr]
        return result

    def _default_eval_metric(self, batch: Any) -> Dict[str, Any]:
        # Best-effort default: treat batch as (x, target, ...) and model as
        # returning a tuple whose first element is the prediction.
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        loss = nn.functional.mse_loss(out, target)
        return {"loss": loss}

    def step_scheduler(self) -> None:
        """Step the LR scheduler.  Call once per epoch."""
        if self.scheduler is not None:
            self.scheduler.step()

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 1,
        eval_metric: Optional[Callable[[nn.Module, Any, torch.device], Dict[str, Any]]] = None,
        checkpoint_path: Optional[str] = None,
        save_best: bool = True,
    ) -> List[Dict[str, Any]]:
        """Train for ``epochs`` epochs and optionally evaluate/ checkpoint.

        Returns the per-epoch history list.
        """
        best_metric = float("inf")
        for _ in range(epochs):
            self.epoch += 1
            if hasattr(self, "model"):
                self.model.epoch = self.epoch
            train_metrics = self.train_epoch(train_loader)
            entry = {"epoch": self.epoch, "train": train_metrics}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, compute_metric=eval_metric)
                entry["val"] = val_metrics
                self.history.append(entry)
                if save_best and checkpoint_path is not None:
                    val_loss = val_metrics.get("loss", float("inf"))
                    if val_loss < best_metric:
                        best_metric = val_loss
                        self.save_checkpoint(checkpoint_path)
                print(
                    f"Epoch {self.epoch}: train_loss={train_metrics.get('loss', float('nan')):.6f}, "
                    f"val_loss={val_metrics.get('loss', float('nan')):.6f}, "
                    f"val_MPJPE={val_metrics.get('mpjpe', float('nan')) * 1000:.2f}mm"
                )
            else:
                self.history.append(entry)
                print(f"Epoch {self.epoch}: train_loss={train_metrics.get('loss', float('nan')):.6f}")
            self.step_scheduler()
        return self.history

    def save_checkpoint(self, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save model, optimizer, scheduler, AMP, and EMA state."""
        state: Dict[str, Any] = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "amp": self.amp.state_dict(),
            "history": self.history,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if extra is not None:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str, *, strict: bool = True) -> Dict[str, Any]:
        """Load a checkpoint saved by ``save_checkpoint``."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.epoch = state.get("epoch", 0)
        self.model.load_state_dict(state["model"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        self.amp.load_state_dict(state.get("amp", {}))
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
        self.history = state.get("history", [])
        return state


# ---------------------------------------------------------------------------
# Convenience trainer for the common multi-view pose model signature
# ---------------------------------------------------------------------------

class MultiViewPoseTrainerV2(TrainerV2):
    """TrainerV2 pre-wired for the common ``(x, y, K, R, t)`` model convention.

    Most ray-attention fusion models in this project accept a 5D input
    ``(B, T, V, J, 3)`` of ``(x_pixel, y_pixel, confidence)`` and camera
    parameters ``K, R, t``, and return a tuple ``(pred_3d, ...)``. This trainer
    assumes batches are tuples ``(x, y, K, R, t)`` and uses ``nn.MSELoss``
    unless another criterion is supplied.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        *,
        criterion: Optional[nn.Module] = None,
        extract_pred: Callable[[Any], torch.Tensor] = lambda out: out[0] if isinstance(out, (list, tuple)) else out,
        **kwargs: Any,
    ):
        super().__init__(model, optimizer, device, compute_loss=None, **kwargs)
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.extract_pred = extract_pred

    def compute_loss(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, y, K, R, t = batch
        x = x.to(device)
        y = y.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        out = model(x, K=K, R=R, t=t)
        pred = self.extract_pred(out)
        loss = self.criterion(pred, y)
        return loss, {}

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_metric: Optional[Callable[[nn.Module, Any, torch.device], Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        if compute_metric is None:
            compute_metric = self._eval_metric
        return super().evaluate(dataloader, compute_metric=compute_metric)

    def _eval_metric(self, model: nn.Module, batch: Any, device: torch.device) -> Dict[str, Any]:
        x, y, K, R, t = batch
        x = x.to(device)
        y = y.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        out = model(x, K=K, R=R, t=t)
        pred = self.extract_pred(out)
        loss = nn.functional.mse_loss(pred, y)
        mpjpe = (pred - y).norm(dim=-1).mean()
        return {"loss": loss, "mpjpe": mpjpe}
