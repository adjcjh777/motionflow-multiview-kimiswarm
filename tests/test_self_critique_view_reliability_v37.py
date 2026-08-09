import torch

from motionflow_mv.fusion.self_critique_view_reliability_v37 import (
    SelfCritiqueViewReliabilityV37,
)


def test_self_critique_view_reliability_v37():
    B, T, V, J, d = 2, 3, 4, 17, 64
    tokens = torch.randn(B, T, V, J, d)
    module = SelfCritiqueViewReliabilityV37(d=d, hidden_dim=32, n_layers=2)
    reliability, view_reliability = module(tokens)
    assert reliability.shape == (B, T, V, J)
    assert view_reliability.shape == (B, T, V)
    assert (reliability >= 0).all() and (reliability <= 1).all()
    assert (view_reliability >= 0).all() and (view_reliability <= 1).all()


def test_self_critique_view_reliability_v37_with_mask():
    B, T, V, J, d = 2, 3, 4, 17, 64
    tokens = torch.randn(B, T, V, J, d)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    view_mask[:, :, 0] = False
    module = SelfCritiqueViewReliabilityV37(d=d, hidden_dim=32, n_layers=2)
    reliability, view_reliability = module(tokens, view_mask=view_mask)
    assert reliability.shape == (B, T, V, J)
    assert view_reliability.shape == (B, T, V)
    assert (reliability[..., 0, :] == 0).all()
