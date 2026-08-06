"""CPU sanity test for the deeper residual-gated temporal attention model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_deeper_temporal_model import (
    RayAttentionFusionModelHierarchicalViewDeeperTemporalResidualPrincipalPoint,
)


def make_data(batch=1, t=3, v=2, j=17):
    x = torch.randn(batch, t, v, j, 3)
    x[..., 2] = (x[..., 2] + 1).clamp(min=0)
    y = torch.randn(batch, t, j, 3)
    K = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    R = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    t_vec = torch.zeros(v, 3).float()
    return x, y, K, R, t_vec


def test_deeper_temporal_forward_backward():
    x, y, K, R, t_vec = make_data()
    model = RayAttentionFusionModelHierarchicalViewDeeperTemporalResidualPrincipalPoint(
        j=17, d=32, n_views=2, n_st_layers=1, residual_hidden=64,
        n_temporal_layers=4, n_view_layers=1, n_view_groups=2,
        n_joint_graph_layers=1, return_pp_delta=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(x, K=K, R=R, t=t_vec)[0]
    loss = (pred - y).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"deeper_temporal_pp: loss={loss.item():.4f}, grads ok")


def test_deeper_temporal_block_shape():
    from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_deeper_temporal_model import (
        _DeeperTemporalBlock,
    )
    block = _DeeperTemporalBlock(d=32, n_layers=3, n_heads=4)
    x = torch.randn(2, 13, 32)
    y = block(x)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_deeper_temporal_block_shape()
    test_deeper_temporal_forward_backward()
    print("deeper_temporal_pp tests passed.")
