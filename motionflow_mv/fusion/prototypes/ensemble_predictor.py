"""Light-weight ensemble inference across multiple checkpoints.

``MultiCheckpointEnsemble`` loads a collection of checkpoints into copies of a
shared model and averages their 3-D pose predictions.  The design is
intentionally small: it re-uses the existing single-checkpoint evaluation logic
and only adds checkpoint aggregation.
"""

from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class MultiCheckpointEnsemble(nn.Module):
    """Run the same model architecture with several checkpoints and ensemble.

    Parameters
    ----------
    build_fn:
        Callable that returns a fresh model instance.  It is invoked once per
        checkpoint.  The callable must accept no arguments (wrap the builder
        with :pyfunc:`functools.partial` if necessary).
    checkpoint_paths:
        Paths to the checkpoints to load.  Order matters when ``weights`` is
        supplied.
    device:
        Target device for every model.
    weights:
        Optional positive weights for a weighted average.  If ``None``, a
        uniform mean is used.  Weights are normalised to sum to one.
    output_index:
        Which element of each model's output tuple to ensemble.  Most models in
        this repo return ``(pred_3d, ...)``, so the default ``0`` is correct.

    Attributes
    ----------
    models: List[nn.Module]
        Loaded models, one per checkpoint.
    """

    def __init__(
        self,
        build_fn: Callable[[], nn.Module],
        checkpoint_paths: Sequence[str],
        device: Union[str, torch.device] = "cpu",
        weights: Optional[Sequence[float]] = None,
        output_index: int = 0,
    ):
        super().__init__()
        if not checkpoint_paths:
            raise ValueError("At least one checkpoint path is required.")

        self.device = torch.device(device)
        self.output_index = output_index

        models: List[nn.Module] = []
        for path in checkpoint_paths:
            model = build_fn().to(self.device)
            state = torch.load(path, map_location=self.device, weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"Warning: missing keys in {path}: {missing[:5]}")
            if unexpected:
                print(f"Warning: unexpected keys in {path} (ignored): {unexpected[:5]}")
            model.eval()
            models.append(model)

        self.models = nn.ModuleList(models)

        if weights is not None:
            weights = torch.as_tensor(list(weights), dtype=torch.float32, device=self.device)
            if len(weights) != len(self.models):
                raise ValueError("Number of weights must match number of checkpoints.")
            if weights.sum().item() <= 0:
                raise ValueError("Sum of ensemble weights must be positive.")
            self.register_buffer("weights", weights / weights.sum())
        else:
            self.register_buffer("weights", torch.ones(len(self.models), device=self.device) / len(self.models))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Run each model and return the (weighted) averaged prediction.

        Returns
        -------
        pred_3d:
            Averaged 3-D pose tensor of shape ``(B, T, J, 3)`` (or whatever
            shape the underlying model returns for its first output).
        """
        preds = []
        with torch.no_grad():
            for model in self.models:
                out = model(*args, **kwargs)
                if isinstance(out, (list, tuple)):
                    out = out[self.output_index]
                preds.append(out)

        stacked = torch.stack(preds, dim=0)  # (M, ...)
        weights = self.weights.view(-1, *([1] * (stacked.dim() - 1)))
        return (stacked * weights).sum(dim=0)

    def predict_all(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both individual predictions and the ensemble average.

        Returns
        -------
        ensemble_pred:
            Weighted mean prediction, same shape as a single model output.
        individual_preds:
            Stack of every model's prediction, ``(M, ...)``.
        """
        preds = []
        with torch.no_grad():
            for model in self.models:
                out = model(*args, **kwargs)
                if isinstance(out, (list, tuple)):
                    out = out[self.output_index]
                preds.append(out)

        stacked = torch.stack(preds, dim=0)
        weights = self.weights.view(-1, *([1] * (stacked.dim() - 1)))
        ensemble = (stacked * weights).sum(dim=0)
        return ensemble, stacked


def load_ensemble_from_checkpoints(
    build_fn: Callable[[], nn.Module],
    checkpoint_paths: Sequence[str],
    device: Union[str, torch.device] = "cpu",
    weights: Optional[Sequence[float]] = None,
) -> MultiCheckpointEnsemble:
    """Convenience helper that constructs a ``MultiCheckpointEnsemble``.

    This mirrors the single-checkpoint ``torch.load`` + ``load_state_dict``
    pattern but across many checkpoints.
    """
    return MultiCheckpointEnsemble(build_fn, checkpoint_paths, device=device, weights=weights)
