import torch
from motionflow_mv.fusion.physical_space_alignment_v28 import PhysicalSpaceAlignmentV28, floor_loss, bone_temporal_loss

def test_identity():
    head = PhysicalSpaceAlignmentV28(j=17)
    X = torch.randn(2, 3, 17, 3)
    assert torch.allclose(head(X), X, atol=1e-5)

def test_reg_loss():
    head = PhysicalSpaceAlignmentV28(j=17)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3)
    out, reg = head(X, return_reg_loss=True)
    assert out.shape == X.shape
    assert reg >= 0.0

def test_residual_bound():
    head = PhysicalSpaceAlignmentV28(j=17, max_residual=0.05)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3)
    assert (head(X) - X).abs().max().item() <= 0.05 * 1.01

def test_floor_robust():
    X = torch.zeros(1, 1, 17, 3)
    X[0, 0, [3, 6, 11], 1] = 0.0
    X[0, 0, 14, 1] = -1.0
    assert 0 <= floor_loss(X, 0.0, [3, 6, 11, 14], floor_quantile=0.25) < 0.5

def test_bone_temporal():
    X = torch.randn(2, 5, 17, 3)
    assert bone_temporal_loss(X, list(range(-1, 16))) >= 0.0
