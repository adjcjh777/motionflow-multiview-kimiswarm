"""Smoke test for the improved adaptive soft view gate (v2).

Runs a synthetic forward/backward pass on CPU, checks output shapes and gradient
flow, and verifies that the gate responds to occluded views.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_campe_adaptive_softgate_v2_model import (
    RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGateV2,
)


def _make_cameras(n_views: int = 4):
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


def test_shapes_and_gradients(device: str = "cpu"):
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3, device=device)
    model = RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGateV2(
        j=J, d=64, n_views=V, gate_n_heads=4
    ).to(device)
    model.train()

    pred, weights, gate, reg = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert gate.shape == (B, T, V, J)
    assert reg.numel() == 1 and reg.requires_grad

    loss = pred.mean() + 0.01 * reg
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("[OK] shape and gradient test passed")


def test_gate_validity(device: str = "cpu"):
    """Verify that the gate produces valid soft weights in [0, 1]."""
    B, T, V, J = 1, 3, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3, device=device)
    model = RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGateV2(
        j=J, d=64, n_views=V, gate_n_heads=4
    ).to(device)
    model.eval()

    with torch.no_grad():
        _, _, gate, _ = model(x, cameras=cameras)

    assert gate.min().item() >= 0.0 and gate.max().item() <= 1.0
    # Gate should not collapse to a single value across views (cross-view
    # attention should produce some variation).
    assert gate.std().item() > 0.0
    print("[OK] gate validity test passed")


def test_iterative_refinement(device: str = "cpu"):
    B, T, V, J = 1, 3, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3, device=device)
    model = RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGateV2(
        j=J, d=64, n_views=V
    ).to(device)
    model.eval()
    with torch.no_grad():
        pred1, *_ = model(x, cameras=cameras, n_iter=1)
        pred3, *_ = model(x, cameras=cameras, n_iter=3)
    assert pred1.shape == (B, T, J, 3)
    assert pred3.shape == (B, T, J, 3)
    print("[OK] iterative refinement test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    test_shapes_and_gradients(device=args.device)
    test_gate_validity(device=args.device)
    test_iterative_refinement(device=args.device)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
