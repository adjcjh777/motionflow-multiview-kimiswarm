"""Model wrapper that applies multi-view training-time augmentation.

This module provides :class:`MultiViewDataAugmentationWrapper`, a thin
``nn.Module`` that wraps an existing fusion/regression model and applies
view-synchronised temporal jitter before forwarding the inputs.  In evaluation
mode the augmentation is disabled, so checkpoints remain compatible with the
wrapped model.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from motionflow_mv.data.multiview_temporal_jitter import MultiViewSyncAugmentation


class MultiViewDataAugmentationWrapper(nn.Module):
    """Wrap a base fusion model with view-synced temporal augmentation.

    During training, inputs are jittered with :class:`MultiViewSyncAugmentation`
    before being passed to ``base_model``.  During evaluation, the augmentation is
    bypassed.  State-dict operations are delegated to the wrapped model so that
    saved checkpoints can be loaded directly into an unwrapped model of the same
    type.

    Parameters
    ----------
    base_model:
        The model to wrap.  Must accept ``x`` as its first positional argument.
    aug_config:
        Dictionary of keyword arguments forwarded to :class:`MultiViewSyncAugmentation`.
        If ``None``, default values are used.
    """

    def __init__(self, base_model: nn.Module, aug_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.base_model = base_model
        self.aug = MultiViewSyncAugmentation(**(aug_config or {}))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        if self.training:
            x = self.aug(x)
        return self.base_model(x, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Delegate to the wrapped model so checkpoint keys remain unchanged."""
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Delegate to the wrapped model."""
        return self.base_model.load_state_dict(state_dict, *args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(base_model={self.base_model}, aug={self.aug})"
