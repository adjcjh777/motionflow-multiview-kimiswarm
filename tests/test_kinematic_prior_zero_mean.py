"""Unit tests for the zero-mean KinematicAnthropometricPrior (KAP).

A zero-mean prior is obtained by setting ``bone_mu`` to zero.  In this
regime the bone-length NLL is minimized for zero-length bones and grows
monotonically as bones get longer.
"""

import torch

from motionflow_mv.fusion.kinematic_anthropometric_prior_v22 import (
    KinematicAnthropometricPrior,
)


def test_zero_mean_prior_penalizes_nonzero_bone_lengths():
    """When bone_mu=0, zero-length bones yield the minimum possible NLL."""
    model = KinematicAnthropometricPrior(
        j=17,
        d=32,
        use_angle_limit=False,
    )
    # Force a zero-mean prior with unit variance for deterministic loss.
    model.bone_mu.data.zero_()
    model.bone_logvar.data.zero_()

    n, j, d = 2, 17, 32
    feat = torch.zeros(n, j, d)

    # All joints at the origin -> every bone length is zero.
    pred_zero = torch.zeros(n, j, 3)
    _, loss_zero = model(feat, pred_zero)

    # Random pose has non-zero bone lengths on average.
    torch.manual_seed(0)
    pred_nonzero = torch.randn(n, j, 3) * 0.3
    _, loss_nonzero = model(feat, pred_nonzero)

    # Zero-length bones are the optimum of a zero-mean Gaussian.
    assert torch.allclose(loss_zero, torch.tensor(0.0), atol=1e-6)
    assert loss_nonzero.item() > loss_zero.item()
