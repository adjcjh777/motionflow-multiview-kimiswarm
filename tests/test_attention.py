"""Unit tests for the lightweight view-attention fusion module."""

import torch

from motionflow_mv.fusion.attention import ViewAttentionFusion
from motionflow_mv.fusion.attention_model import AttentionFusionModel


def test_view_attention_fusion_shape():
    module = ViewAttentionFusion(d=32, j=17)
    x = torch.randn(2, 4, 17, 32)
    y = module(x)
    assert y.shape == (2, 17, 32)


def test_attention_fusion_model_shape():
    model = AttentionFusionModel(j=17, d=32, n_views=4)
    x = torch.randn(2, 4, 17, 3)
    y = model(x)
    assert y.shape == (2, 17, 3)


if __name__ == "__main__":
    test_view_attention_fusion_shape()
    test_attention_fusion_model_shape()
    print("attention tests passed")
