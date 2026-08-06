"""CPU sanity test: one forward/backward step for each iter14 model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_graph_skeleton_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar,
)


def make_data(batch=1, t=3, v=2, j=17):
    x = torch.randn(batch, t, v, j, 3)
    x[..., 2] = (x[..., 2] + 1).clamp(min=0)  # confidences >= 0
    y = torch.randn(batch, t, j, 3)
    K = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    R = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    t_vec = torch.zeros(v, 3).float()
    return x, y, K, R, t_vec


def test_model(cls, name, **kwargs):
    print(f"Testing {name}...")
    x, y, K, R, t_vec = make_data()
    model = cls(j=17, d=32, n_views=2, n_st_layers=1, residual_hidden=64, return_pp_delta=True, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(x, K=K, R=R, t=t_vec)[0]
    loss = (pred - y).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  {name}: loss={loss.item():.4f}, grads ok")


def main():
    test_model(RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint, "pp_baseline")
    test_model(RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate, "dynamic_gate", return_gate=True)
    test_model(RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual, "graph_skeleton_residual")
    test_model(RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar, "epipolar")
    print("All iter14 models passed one train step on CPU.")


if __name__ == "__main__":
    main()
