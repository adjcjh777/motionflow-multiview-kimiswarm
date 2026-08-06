"""CPU smoke tests for the physics-informed skeleton dynamics prior.

Verifies that the temporal skeleton-dynamics refiner model and its auxiliary
physics loss can be instantiated, forward-propagated, and back-propagated
without crashing on synthetic inputs.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_physics_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointPhysics,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _make_cameras
from motionflow_mv.fusion.graph_joint_relation import H36M_17_PARENTS
from motionflow_mv.losses.physics_informed_dynamics import PhysicsInformedSkeletonDynamicsLoss


def test_physics_model_forward_and_backward():
    B, T, V, J = 2, 9, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    y = torch.rand(B, T, J, 3)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointPhysics(
        j=J, d=32, n_views=V, n_st_layers=2, residual_hidden=64,
        dynamics_hidden=64, return_raw=True,
    )
    pred, weights, raw = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert raw.shape == (B, T, J, 3)

    phys_loss_fn = PhysicsInformedSkeletonDynamicsLoss(
        parents=H36M_17_PARENTS,
        foot_indices=[3, 6, 10, 13],
    )
    loss = torch.nn.functional.mse_loss(pred, y) + 0.01 * phys_loss_fn(pred)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_physics_model_single_frame():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointPhysics(
        j=J, d=32, n_views=V, n_st_layers=2, residual_hidden=64,
        dynamics_hidden=64,
    )
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


def test_physics_loss_independent():
    pred = torch.rand(2, 11, 17, 3)
    phys_loss_fn = PhysicsInformedSkeletonDynamicsLoss(
        parents=H36M_17_PARENTS,
        foot_indices=[3, 6, 10, 13],
    )
    loss = phys_loss_fn(pred)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_physics_loss_independent()
    test_physics_model_forward_and_backward()
    test_physics_model_single_frame()
    print("physics dynamics prior smoke tests passed")
