"""Batch routing for the multi-person association graph."""

import torch

from motionflow_mv.fusion.multiperson_association_graph import MultiPersonAssociationGraph


def test_batched_graph_matches_independent_second_sample():
    torch.manual_seed(0)
    graph = MultiPersonAssociationGraph(
        d=4,
        n_views=1,
        n_persons=1,
        num_layers=1,
    ).eval()
    x = torch.randn(2, 1, 1, 2, 4)

    with torch.no_grad():
        batched = graph(x, parents=[-1, 0], symmetry_pairs=[])
        independent = graph(x[1:], parents=[-1, 0], symmetry_pairs=[])

    torch.testing.assert_close(batched[1], independent[0])
