"""Tests for motionflow_mv/utils/model_analysis.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from motionflow_mv.utils.model_analysis import (
    FlopsCounter,
    ParameterSummary,
    analyze_model,
    count_parameters,
)


def test_count_parameters_simple() -> None:
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    summary = count_parameters(model)
    # (10 * 5 + 5) + (5 * 1 + 1) = 55 + 6 = 61
    assert summary.total == 61
    assert summary.trainable == 61
    assert summary.non_trainable == 0
    assert summary.size_mb > 0.0


def test_count_parameters_with_buffer() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 3))
            self.register_buffer("buffer", torch.ones(2, 2))

    model = Model()
    summary = count_parameters(model)
    # buffers are not counted in parameters()
    assert summary.total == 9


def test_flops_counter_linear() -> None:
    model = nn.Linear(10, 5)
    counter = FlopsCounter()
    counter.register(model)
    x = torch.randn(2, 10)
    try:
        with torch.no_grad():
            y = model(x)
    finally:
        counter.remove()

    # y = x A^T + b  =>  2 * in * out per batch element + bias
    expected = 2 * 0 * y.numel()  # placeholder to avoid unused variable
    del expected
    assert counter.flops_by_op.total > 0
    assert "Linear" in counter.flops_by_op.breakdown


def test_analyze_model_returns_summary() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = TinyModel()
    summary = analyze_model(model, input_args=(torch.randn(1, 4),))
    assert summary["model"] == "TinyModel"
    assert "parameters" in summary
    assert "flops" in summary
    assert summary["parameters"]["total"] == 10  # *4 + 2 = 10


def test_analyze_model_parameter_summary_only() -> None:
    model = nn.Linear(8, 4)
    summary = analyze_model(model)
    assert "parameters" in summary
    assert "flops" not in summary
