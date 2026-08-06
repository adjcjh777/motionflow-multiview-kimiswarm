"""CPU smoke test for the Bayesian triangulation ablation flags."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri,
)


def _make_cameras(n_views: int = 4):
    import numpy as np
    from motionflow_mv.calibration.camera import Camera
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def test_bayesian_tri_ablation_flags():
    B, T, V, J = 2, 3, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    y = torch.rand(B, T, J, 3)
    for anisotropic in [True, False]:
        for use_adaptive_gn in [True, False]:
            model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri(
                j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64,
                anisotropic_covariance=anisotropic, use_adaptive_gn=use_adaptive_gn,
            )
            pred, weights, pp_delta, epi_loss = model(x, cameras=cameras)
            assert pred.shape == (B, T, J, 3)
            assert weights.shape == (B, T, V, J)
            assert pp_delta.shape == (B * T, V, 2)
            assert epi_loss.shape == ()
            loss = (pred - y).pow(2).mean() + 0.0 * epi_loss
            loss.backward()
            assert any(p.grad is not None for p in model.parameters())
            expected_cov_out = 3 if anisotropic else 1
            assert model.covariance_head[-1].out_features == expected_cov_out
    print("All ablation flag combinations passed forward/backward on CPU.")


if __name__ == "__main__":
    test_bayesian_tri_ablation_flags()
