"""CPU smoke test for the deeper st attention prototype.

Usage
-----
    python scripts/smoke_deeper_st_attention_cpu.py

The script checks that ``DeeperStAttentionPrincipalPointModel``:
    - accepts both 4-D and 5-D inputs,
    - returns the expected output shapes,
    - computes gradients end-to-end,
    - and reports a stable parameter count on CPU.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.prototypes.deeper_st_attention_model.deeper_st_attention_model import (  # noqa: E501
    DeeperStAttentionPrincipalPointModel,
)
from motionflow_mv.models.spatiotemporal_principal_point_model import _make_cameras


def _run_smoke(batch_temporal: bool):
    B = 2
    T = 5 if batch_temporal else 1
    V = 4
    J = 17

    cameras = _make_cameras(V)
    if batch_temporal:
        x = torch.rand(B, T, V, J, 3)
    else:
        x = torch.rand(B, V, J, 3)

    model = DeeperStAttentionPrincipalPointModel(
        j=J,
        d=64,
        n_views=V,
        n_st_blocks=3,
        return_pp_delta=True,
    )

    pred, weights, pp_delta = model(x, cameras=cameras)

    expected_pred_shape = (B, T, J, 3) if batch_temporal else (B, J, 3)
    expected_weight_shape = (B, T, V, J) if batch_temporal else (B, V, J)

    assert pred.shape == expected_pred_shape, f"pred shape mismatch: {pred.shape}"
    assert weights.shape == expected_weight_shape, f"weights shape mismatch: {weights.shape}"
    assert pp_delta.shape == (B, T, V, 2), f"pp_delta shape mismatch: {pp_delta.shape}"

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters()), "no gradients computed"

    param_count = sum(p.numel() for p in model.parameters())
    return param_count


def main():
    torch.manual_seed(42)

    param_count_4d = _run_smoke(batch_temporal=False)
    print(f"4-D input smoke test passed, params={param_count_4d:,}")

    param_count_5d = _run_smoke(batch_temporal=True)
    print(f"5-D input smoke test passed, params={param_count_5d:,}")

    print("All deeper st attention CPU smoke tests passed.")


if __name__ == "__main__":
    main()
