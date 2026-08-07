"""CPU smoke tests for synchronized multi-view 2D augmentation."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.data.sync_multiview_aug import (
    SynchronizedMultiview2DAugmenter,
    flip_horizontal,
    rotate,
    scale,
    translate,
)


def test_flip_horizontal_preserves_views():
    V, J, C = 4, 17, 3
    x = torch.arange(100, 100 + V * J * C).float().view(V, J, C)
    x[..., 2] = 1.0
    out = flip_horizontal(x.clone(), image_size=(640, 480))
    assert out.shape == x.shape
    assert torch.allclose(out[..., 2], x[..., 2])
    # Every view should be flipped around the same axis.
    assert torch.allclose(out[..., 0], 640.0 - x[..., 0])


def test_rotate_90_around_origin():
    x = torch.tensor([[[[1.0, 0.0, 1.0]]]])  # (1, 1, 1, 3)
    out = rotate(x, angle_deg=90.0)
    assert out.shape == x.shape
    # (1, 0) rotated 90 CCW around origin -> (0, 1)
    assert torch.allclose(out[..., :2], torch.tensor([[[[0.0, 1.0]]]]), atol=1e-5)


def test_scale_and_translate_shape():
    B, V, J, C = 2, 4, 17, 3
    x = torch.rand(B, V, J, C)
    out = scale(x, scale_factor=1.1, image_size=(640, 480))
    assert out.shape == x.shape
    out = translate(x, dx=5.0, dy=-3.0)
    assert out.shape == x.shape


def test_augmenter_synchronizes_across_views():
    """All views in a sample should receive the exact same transformation."""
    torch.manual_seed(0)
    B, V, J, C = 2, 4, 17, 3
    x = torch.rand(B, V, J, C) * 100.0
    x[..., 2] = 1.0

    augmenter = SynchronizedMultiview2DAugmenter(
        horizontal_flip_prob=0.0,
        rotation_deg=0.0,
        scale_range=(1.0, 1.0),
        translation_px=10.0,
        image_size=(640, 480),
        per_sample=False,
        seed=42,
    )
    out = augmenter(x)
    assert out.shape == x.shape

    # With only translation applied, the per-view delta should be preserved.
    delta_in = x[:, 0:1, :, :2] - x[:, 1:2, :, :2]
    delta_out = out[:, 0:1, :, :2] - out[:, 1:2, :, :2]
    assert torch.allclose(delta_in, delta_out, atol=1e-4)


def test_augmenter_state_dict_roundtrip():
    augmenter = SynchronizedMultiview2DAugmenter(seed=123)
    state = augmenter.state_dict()
    augmenter.generator.manual_seed(999)  # perturb
    augmenter.load_state_dict(state)
    assert augmenter.horizontal_flip_prob == state["horizontal_flip_prob"]
    assert augmenter.scale_range == tuple(state["scale_range"])


def test_per_sample_parameters_differ():
    """With per_sample=True, different batch elements should be transformed differently."""
    torch.manual_seed(0)
    B, V, J, C = 4, 4, 17, 3
    x = torch.rand(B, V, J, C) * 100.0
    x[..., 2] = 1.0

    augmenter = SynchronizedMultiview2DAugmenter(
        horizontal_flip_prob=0.5,
        rotation_deg=30.0,
        scale_range=(0.8, 1.2),
        translation_px=20.0,
        image_size=(640, 480),
        per_sample=True,
        seed=7,
    )
    out = augmenter(x)
    # Different batch elements should not all be identical (highly likely given
    # the random parameter ranges and 4 batch elements).
    any_different = False
    for i in range(1, B):
        if not torch.allclose(out[0], out[i]):
            any_different = True
            break
    assert any_different, "per_sample=True should produce different transformations per sample"


if __name__ == "__main__":
    test_flip_horizontal_preserves_views()
    test_rotate_90_around_origin()
    test_scale_and_translate_shape()
    test_augmenter_synchronizes_across_views()
    test_augmenter_state_dict_roundtrip()
    test_per_sample_parameters_differ()
    print("synchronized multiview 2D augmentation tests passed")
