"""CPU smoke tests for the cross-view graph attention fusion prototype.

This module does **not** start any GPU training; it only checks that the
prototype module can be instantiated, run a forward pass, and produce
gradients.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    CrossViewGraphAttention,
    CrossViewGraphAttentionLayer,
)
from motionflow_mv.fusion.graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    build_edge_index,
)


def test_layer_forward_28j():
    B, V, J, d = 2, 4, 28, 64
    x = torch.rand(B, V, J, d)
    edge_index, edge_type = build_edge_index(
        MPI_INF_3DHP_28_PARENTS, MPI_INF_3DHP_28_SYMMETRY_PAIRS, V, J
    )

    layer = CrossViewGraphAttentionLayer(d=d, n_heads=4)
    out = layer(x, edge_index, edge_type)
    assert out.shape == (B, V, J, d)

    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in layer.parameters())


def test_layer_forward_17j():
    B, V, J, d = 1, 4, 17, 32
    x = torch.rand(B, V, J, d)
    edge_index, edge_type = build_edge_index(
        H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, V, J
    )

    layer = CrossViewGraphAttentionLayer(d=d, n_heads=4)
    out = layer(x, edge_index, edge_type)
    assert out.shape == (B, V, J, d)



def test_stack_forward():
    B, V, J, d = 2, 4, 17, 64
    x = torch.rand(B, V, J, d)

    model = CrossViewGraphAttention(d=d, n_views=V, n_layers=2, n_heads=4)
    model.build_edge_index(J, H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS)

    out = model(x)
    assert out.shape == (B, V, J, d)

    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_cross_view_edges_reachability():
    """Ensure the same joint can attend across views via cross-view edges."""
    V, J = 4, 17
    edge_index, edge_type = build_edge_index(
        H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, V, J
    )

    # Edge type 2 is cross-view; there should be at least one edge between
    # each pair of views for every joint.
    cross_view = edge_type == 2
    assert cross_view.sum().item() == V * (V - 1) * J


if __name__ == "__main__":
    test_layer_forward_28j()
    print("test_layer_forward_28j passed")
    test_layer_forward_17j()
    print("test_layer_forward_17j passed")
    test_stack_forward()
    print("test_stack_forward passed")
    test_cross_view_edges_reachability()
    print("test_cross_view_edges_reachability passed")
    print("All cross-view graph attention smoke tests passed")
