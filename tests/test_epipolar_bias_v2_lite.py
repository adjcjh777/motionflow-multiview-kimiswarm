"""CPU smoke test for the late-layer epipolar-bias v2 lite model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.epipolar_bias_v2_lite_pp_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2Lite,
)


def _synthetic_clip(batch=2, t=5, v=4, j=17):
    x = torch.randn(batch, t, v, j, 3)
    x[..., 2] = (x[..., 2] + 1).clamp(min=0.0)
    y = torch.randn(batch, t, j, 3)
    K = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    R = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    t_vec = torch.zeros(v, 3).float()
    return x, y, K, R, t_vec


def test_synthetic_forward_backward():
    x, y, K, R, t_vec = _synthetic_clip()
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2Lite(
        j=17, d=32, n_views=4, n_st_layers=2, residual_hidden=64, return_pp_delta=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    pred, weights, pp_delta = model(x, K=K, R=R, t=t_vec)
    loss = (pred - y).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[synthetic] pred={tuple(pred.shape)}, weights={tuple(weights.shape)}, loss={loss.item():.4f}")


def test_smoke_npz_one_step():
    import numpy as np
    npz_path = Path("tmp/mpi_s01_seq01_smoke.npz")
    if not npz_path.exists():
        print(f"[smoke npz] {npz_path} not found; skipping")
        return
    data = np.load(npz_path)
    points_2d = torch.from_numpy(data["points_2d"]).float()[:13]
    confidences = torch.from_numpy(data["confidences"]).float()[:13]
    joints_3d = torch.from_numpy(data["joints_3d"]).float()[:13]
    K = torch.from_numpy(data["camera_K"]).float()
    R = torch.from_numpy(data["camera_R"]).float()
    t = torch.from_numpy(data["camera_t"]).float()
    x = torch.cat([points_2d, confidences.unsqueeze(-1)], dim=-1).unsqueeze(0)
    y = joints_3d.unsqueeze(0)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2Lite(
        j=points_2d.shape[2], d=32, n_views=K.shape[0], n_st_layers=2, residual_hidden=64, return_pp_delta=True
    )
    model.eval()
    with torch.no_grad():
        pred = model(x, K=K, R=R, t=t)[0]
    print(f"[smoke npz] pred={tuple(pred.shape)}, gt={tuple(y.shape)}, gate={torch.sigmoid(model.epipolar_gate).item():.4f}")


def main():
    test_synthetic_forward_backward()
    test_smoke_npz_one_step()
    print("epipolar_bias_v2_lite_pp smoke test passed")


if __name__ == "__main__":
    main()
