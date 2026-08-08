"""Lightweight model size and FLOP analysis utilities.

This module deliberately avoids heavy third-party profiling packages so it can
run on the local CPU-only workstation while the A800 GPUs are busy.  It provides:

* Parameter counting (total / trainable / non-trainable, model size in MB).
* A tiny hook-based FLOP estimator for common layers (``Linear``, ``Conv*``,
  ``BatchNorm``, ``LayerNorm``, activations, ``MatMul``-like operations).
* A high-level ``analyze_model`` helper that returns a JSON-serializable dict.

Notes
-----
The FLOP counts are *approximate* and count multiply-add pairs as one FLOP.
For attention blocks that use ``torch.matmul`` directly, add a custom handler or
use the analytical helpers provided in ``scripts/analyze_model_size_flops.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ParameterSummary:
    """Summary of a model's parameter footprint."""

    total: int = 0
    trainable: int = 0
    non_trainable: int = 0
    size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "non_trainable": self.non_trainable,
            "size_mb": round(self.size_mb, 6),
        }


@dataclass
class FlopsByOp:
    """FLOPs grouped by operation type."""

    total: int = 0
    breakdown: Dict[str, int] = field(default_factory=dict)

    def add(self, op_type: str, flops: int) -> None:
        if flops <= 0:
            return
        self.total += flops
        self.breakdown[op_type] = self.breakdown.get(op_type, 0) + flops

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "breakdown": {k: v for k, v in self.breakdown.items()},
        }


def count_parameters(model: nn.Module) -> ParameterSummary:
    """Count trainable and non-trainable parameters, and estimate float32 size."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    size_mb = total * 4 / (1024 ** 2)  # float32 bytes per param
    return ParameterSummary(
        total=total,
        trainable=trainable,
        non_trainable=non_trainable,
        size_mb=size_mb,
    )


def _linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
    # y = x A^T + b  =>  2 * in_features * out_features per element in batch
    in_features = module.in_features
    out_features = module.out_features
    out_elements = output.numel()
    flops = 2 * in_features * out_elements
    if module.bias is not None:
        flops += out_elements
    return flops


def _conv_flops(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> int:
    # Approximate: 2 * kernel_elements * in_channels * out_channels * output_spatial
    if not hasattr(module, "kernel_size"):
        return 0
    kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
    kernel_elements = 1
    for k in kernel_size:
        kernel_elements *= k
    in_channels = module.in_channels if hasattr(module, "in_channels") else input.shape[1]
    out_channels = module.out_channels if hasattr(module, "out_channels") else output.shape[1]
    output_spatial = output.numel() // (output.shape[0] * out_channels) if len(output.shape) > 2 else 1
    flops = 2 * in_channels * out_channels * kernel_elements * output_spatial
    if getattr(module, "bias", None) is not None:
        flops += output.numel()
    return flops


def _norm_flops(module: nn.Module, output: torch.Tensor) -> int:
    # BatchNorm: ~5 ops per element, LayerNorm: ~5 ops per element
    return 5 * output.numel()


def _activation_flops(module: nn.Module, output: torch.Tensor) -> int:
    # Treat element-wise activations as 1 FLOP per element.
    return output.numel()


class FlopsCounter:
    """Hook-based FLOP counter.

    Register hooks on modules.  During the forward pass the counter estimates
    the FLOPs contributed by each registered module and aggregates them by
    operation type.

    Parameters
    ----------
    custom_handlers:
        Mapping from ``module.__class__`` to a handler function.  The handler
        must accept ``(module, input, output)`` and return an integer FLOP
        count.
    """

    def __init__(self, custom_handlers: Optional[Dict[type, Callable[[nn.Module, Tuple[Any, ...], torch.Tensor], int]]] = None):
        self.custom_handlers = custom_handlers or {}
        self.flops_by_op = FlopsByOp()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _default_handler(self, module: nn.Module, input: Tuple[Any, ...], output: torch.Tensor) -> int:
        if isinstance(module, nn.Linear):
            return _linear_flops(module, output)
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return _conv_flops(module, input[0], output)
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            return _norm_flops(module, output)
        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
            return _activation_flops(module, output)
        return 0

    def _hook_fn(self, module: nn.Module, input: Tuple[Any, ...], output: torch.Tensor) -> None:
        handler = self.custom_handlers.get(module.__class__, self._default_handler)
        flops = handler(module, input, output)
        if flops:
            self.flops_by_op.add(module.__class__.__name__, flops)

    def register(self, model: nn.Module) -> None:
        """Attach forward hooks to every module in ``model``."""
        for m in model.modules():
            handle = m.register_forward_hook(self._hook_fn)
            self._hooks.append(handle)

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "FlopsCounter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()


def analyze_model(
    model: nn.Module,
    input_args: Optional[Tuple[Any, ...]] = None,
    input_kwargs: Optional[Dict[str, Any]] = None,
    custom_handlers: Optional[Dict[type, Callable[[nn.Module, Tuple[Any, ...], torch.Tensor], int]]] = None,
) -> Dict[str, Any]:
    """Analyze ``model`` parameters and (optionally) FLOPs.

    Parameters
    ----------
    model:
        The PyTorch model to analyze.
    input_args:
        Positional arguments to ``model.forward`` for FLOP profiling.
    input_kwargs:
        Keyword arguments to ``model.forward`` for FLOP profiling.
    custom_handlers:
        Optional mapping from module class to FLOP handler.

    Returns
    -------
    dict:
        JSON-serializable summary with ``parameters`` and ``flops`` keys.
    """
    model.eval()
    params = count_parameters(model)
    result: Dict[str, Any] = {
        "model": model.__class__.__name__,
        "parameters": params.to_dict(),
    }

    if input_args is not None or input_kwargs is not None:
        input_args = input_args or ()
        input_kwargs = input_kwargs or {}
        counter = FlopsCounter(custom_handlers=custom_handlers)
        counter.register(model)
        try:
            with torch.no_grad():
                model(*input_args, **input_kwargs)
        finally:
            counter.remove()
        result["flops"] = counter.flops_by_op.to_dict()

    return result


def format_summary(summary: Dict[str, Any]) -> str:
    """Return a human-readable rendering of an analysis summary."""
    lines = [
        f"Model: {summary['model']}",
        f"  Total params:      {summary['parameters']['total']:,}",
        f"  Trainable params:  {summary['parameters']['trainable']:,}",
        f"  Non-trainable:     {summary['parameters']['non_trainable']:,}",
        f"  Model size (fp32): {summary['parameters']['size_mb']:.2f} MB",
    ]
    if "flops" in summary:
        total = summary["flops"]["total"]
        lines.append(f"  FLOPs (hook-based): {total:,}  ({total / 1e9:.3f} GFLOPs)")
        if summary["flops"]["breakdown"]:
            lines.append("  Per-op breakdown:")
            for op, flops in summary["flops"]["breakdown"].items():
                lines.append(f"    {op}: {flops:,} ({flops / 1e9:.3f} GFLOPs)")
    return "\n".join(lines)
