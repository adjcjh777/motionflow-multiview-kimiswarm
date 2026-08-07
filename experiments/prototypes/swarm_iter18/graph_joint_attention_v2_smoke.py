"""CPU smoke tests for ``motionflow_mv.fusion.graph_joint_attention_v2``.

This script does **not** start any GPU training; it only checks that the
module can be instantiated, run a forward pass, and produce gradients.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.graph_joint_attention_v2 import (
    GraphJointAttentionLayer,
    GraphJointAttentionV2,
    build_graph_joint_edge_index,
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)


def test_layer_forward_28j():
    B, V, J, d = 2, 4, 28, 64
    x = torch.rand(B, V, J, d)
    edge_index, edge_type = build_graph_joint_edge_index(
        MPI_INF_3DHP_28_PARENTS, MPI_INF_3DHP_28_SYMMETRY_PAIRS, V, J
    )

    layer = GraphJointAttentionLayer(d=d, n_heads=4, ffn_hidden=d * 2)
    out = layer(x, edge_index, edge_type)
    assert out.shape == (B, V, J, d)

    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in layer.parameters())


def test_layer_forward_17j():
    B, V, J, d = 1, 4, 17, 32
    x = torch.rand(B, V, J, d)
    edge_index, edge_type = build_graph_joint_edge_index(
        H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, V, J
    )

    layer = GraphJointAttentionLayer(d=d, n_heads=4)
    out = layer(x, edge_index, edge_type)
    assert out.shape == (B, V, J, d)


def test_stack_forward_with_ffn():
    B, V, J, d = 2, 4, 17, 64
    x = torch.rand(B, V, J, d)

    model = GraphJointAttentionV2(d=d, n_views=V, n_layers=2, n_heads=4, ffn_hidden=d)
    model.build_edge_index(J, H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS)

    out = model(x)
    assert out.shape == (B, V, J, d)

    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_edge_counts():
    V, J = 4, 17
    edge_index, edge_type = build_graph_joint_edge_index(
        H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, V, J
    )

    # Self-loops: one per (view, joint).
    assert (edge_type == 3).sum().item() == V * J

    # Cross-view edges: each joint links each pair of views in both directions.
    cross_view = edge_type == 2
    assert cross_view.sum().item() == V * (V - 1) * J

    # Bone edges are bidirectional; symmetry edges are bidirectional.
    # Quick sanity: total edge count should be positive and even (because every
    # undirected relationship is represented by two directed edges).
    assert edge_index.shape[1] > 0
    assert edge_index.shape[1] % 2 == 0


def test_variable_view_count():
    B, J, d = 2, 17, 32
    for V in (2, 3, 4):
        x = torch.rand(B, V, J, d)
        edge_index, edge_type = build_graph_joint_edge_index(
            H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, V, J
        )
        layer = GraphJointAttentionLayer(d=d, n_heads=4)
        out = layer(x, edge_index, edge_type)
        assert out.shape == (B, V, J, d)


def test_dropout_path():
    B, V, J, d = 2, 4, 17, 64
    x = torch.rand(B, V, J, d)
    edge_index, edge_type = build_graph_joint_edge_index(
        H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, V, J
    )
    layer = GraphJointAttentionLayer(d=d, n_heads=4, dropout=0.5)
    out = layer(x, edge_index, edge_type)
    assert out.shape == (B, V, J, d)


if __name__ == "__main__":
    test_layer_forward_28j()
    print("test_layer_forward_28j passed")
    test_layer_forward_17j()
    print("test_layer_forward_17j passed")
    test_stack_forward_with_ffn()
    print("test_stack_forward_with_ffn passed")
    test_edge_counts()
    print("test_edge_counts passed")
    test_variable_view_count()
    print("test_variable_view_count passed")
    test_dropout_path()
    print("test_dropout_path passed")
    print("All GraphJointAttentionV2 CPU smoke tests passed")
