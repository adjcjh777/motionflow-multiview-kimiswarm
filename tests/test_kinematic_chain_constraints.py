"""Smoke tests for the kinematic-chain constraints auxiliary loss."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.graph_joint_relation import (
    H36M_17_PARENTS,
    MPI_INF_3DHP_28_PARENTS,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain,
)
from motionflow_mv.losses import kinematic_chain_loss


def _make_cameras(V: int = 4):
    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    R = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    t = torch.zeros(V, 3).float()
    return K, R, t


def test_kinematic_chain_loss_h36m():
    B, J = 4, 17
    pred = torch.randn(B, J, 3)
    target = torch.randn(B, J, 3)
    loss = kinematic_chain_loss(pred, target, H36M_17_PARENTS)
    assert loss.shape == ()
    assert loss.item() >= 0.0
    loss_zero = kinematic_chain_loss(target, target, H36M_17_PARENTS)
    assert loss_zero.item() == 0.0


def test_kinematic_chain_loss_mpi():
    B, J = 4, 28
    pred = torch.randn(B, J, 3)
    target = torch.randn(B, J, 3)
    loss = kinematic_chain_loss(pred, target, MPI_INF_3DHP_28_PARENTS)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_kinematic_chain_model_forward_and_loss():
    B, T, V, J = 2, 13, 4, 28
    x = torch.randn(B, T, V, J, 3)
    y = torch.randn(B, T, J, 3)
    K, R, t = _make_cameras(V)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain(
        j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64, return_pp_delta=True
    )
    pred, weights, pp_delta = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)

    loss = kinematic_chain_loss(pred, y, MPI_INF_3DHP_28_PARENTS)
    loss.backward()
    assert loss.item() >= 0.0


def test_kinematic_chain_loss_temporal_shape():
    B, T, J = 2, 13, 17
    pred = torch.randn(B, T, J, 3)
    target = torch.randn(B, T, J, 3)
    loss = kinematic_chain_loss(pred, target, H36M_17_PARENTS)
    assert loss.shape == ()
