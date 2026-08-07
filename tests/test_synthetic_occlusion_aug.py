"""CPU smoke tests for synthetic occlusion augmentation.

Tests the public API in ``motionflow_mv.data.synthetic_occlusion_aug``.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.synthetic_occlusion_aug import (
    H36M_17_JOINT_GROUPS,
    MPI_INF_3DHP_28_JOINT_GROUPS,
    SyntheticJointOcclusionAugmenter,
    occlude_joint_groups,
    random_occlude_joint_groups,
)


def _make_input(shape, conf=1.0):
    x = torch.ones(*shape)
    x[..., -1] = conf
    return x


def test_occlude_named_groups():
    """Occluding named groups zeroes the confidence channel for the right joints."""
    V, J, C = 4, 17, 3
    x = _make_input((2, V, J, C))

    out = occlude_joint_groups(x, "h36m_17", group_names=["left_arm"])

    left_arm_joints = set(H36M_17_JOINT_GROUPS["left_arm"])
    for j in range(J):
        conf = out[0, 0, j, -1]
        if j in left_arm_joints:
            assert conf.item() == 0.0, f"joint {j} should be occluded"
        else:
            assert conf.item() == 1.0, f"joint {j} should remain visible"


def test_occlude_zero_coords():
    """When zero_coords=True, coordinate channels are also zeroed."""
    V, J, C = 4, 17, 3
    x = _make_input((1, V, J, C))
    out = occlude_joint_groups(x, "h36m_17", group_names=["head"], zero_coords=True)
    head_joints = H36M_17_JOINT_GROUPS["head"]
    assert (out[0, :, head_joints, :2] == 0.0).all()
    assert (out[0, :, head_joints, 2] == 0.0).all()


def test_occlude_custom_dict():
    """A custom dict of groups can be passed directly."""
    groups = {"upper": [0, 1, 2], "lower": [3, 4, 5]}
    x = _make_input((1, 2, 6, 3))
    out = occlude_joint_groups(x, groups, group_names=["upper"])
    assert (out[0, :, [0, 1, 2], -1] == 0.0).all()
    assert (out[0, :, [3, 4, 5], -1] == 1.0).all()


def test_random_occlude_joint_groups():
    """Random group occlusion drops at least one group when rate=1.0."""
    V, J, C = 4, 17, 3
    x = _make_input((10, V, J, C))
    out = random_occlude_joint_groups(x, group_rate=1.0, skeleton="h36m_17")
    # All confidences should be zero because every group is occluded.
    assert (out[..., -1] == 0.0).all()


def test_random_occlude_per_sample():
    """per_sample=True samples independently for each leading sample."""
    V, J, C = 4, 17, 3
    x = _make_input((2, V, J, C))
    out = random_occlude_joint_groups(
        x,
        group_rate=0.0,
        skeleton="h36m_17",
        per_sample=True,
    )
    assert torch.equal(x, out)


def test_augmenter_shape_preserve():
    """The augmenter preserves input shape and does not mutate the input."""
    augmenter = SyntheticJointOcclusionAugmenter(
        skeleton="h36m_17",
        group_rate=0.5,
        joint_rate=0.1,
        seed=42,
    )
    x = torch.rand(2, 4, 17, 3)
    x_clone = x.clone()
    out = augmenter(x)
    assert out.shape == x.shape
    assert torch.equal(x, x_clone)


def test_augmenter_temporal_consistency():
    """With temporal_consistency=True, all frames share the same occlusion mask."""
    augmenter = SyntheticJointOcclusionAugmenter(
        skeleton="h36m_17",
        group_rate=1.0,
        joint_rate=0.0,
        temporal_consistency=True,
        seed=42,
    )
    x = _make_input((3, 7, 4, 17, 3))
    out = augmenter(x)
    # All frames within a clip should be identical after full group occlusion.
    for b in range(out.shape[0]):
        for t in range(1, out.shape[1]):
            assert torch.equal(out[b, 0], out[b, t])


def test_augmenter_mpiinf3dhp():
    """The augmenter supports the MPI-INF-3DHP 28-joint skeleton."""
    augmenter = SyntheticJointOcclusionAugmenter(
        skeleton="mpiinf3dhp_28",
        group_rate=1.0,
        seed=42,
    )
    x = _make_input((1, 4, 28, 3))
    out = augmenter(x)
    # All groups are occluded, so every joint confidence should be zero.
    assert (out[..., -1] == 0.0).all()


def test_state_dict_round_trip():
    """state_dict / load_state_dict round-trip preserves behavior."""
    augmenter = SyntheticJointOcclusionAugmenter(
        skeleton="h36m_17",
        group_rate=0.5,
        joint_rate=0.1,
        temporal_consistency=True,
        zero_coords=True,
        seed=123,
    )
    state = augmenter.state_dict()

    augmenter2 = SyntheticJointOcclusionAugmenter(seed=None)
    augmenter2.load_state_dict(state)

    assert augmenter2.skeleton == "h36m_17"
    assert augmenter2.group_rate == 0.5
    assert augmenter2.joint_rate == 0.1
    assert augmenter2.temporal_consistency is True
    assert augmenter2.zero_coords is True

    x = torch.rand(2, 5, 4, 17, 3)
    torch.manual_seed(0)
    out1 = augmenter(x)
    out2 = augmenter2(x)
    assert torch.equal(out1, out2)


def test_no_group_rate_returns_input_unchanged():
    """When group_rate and joint_rate are zero, output equals input."""
    augmenter = SyntheticJointOcclusionAugmenter(
        skeleton="h36m_17",
        group_rate=0.0,
        joint_rate=0.0,
        seed=42,
    )
    x = torch.rand(2, 4, 17, 3)
    out = augmenter(x)
    assert torch.equal(x, out)


def test_unsupported_skeleton_raises():
    """An unsupported skeleton alias raises ValueError."""
    try:
        SyntheticJointOcclusionAugmenter(skeleton="unsupported_42")
    except ValueError as exc:
        assert "unsupported_42" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported skeleton")


def test_confidence_channel_validation():
    """Out-of-bounds confidence channel raises ValueError."""
    x = _make_input((1, 2, 17, 3))
    try:
        occlude_joint_groups(x, "h36m_17", group_names=["head"], confidence_channel=5)
    except ValueError as exc:
        assert "confidence_channel" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-bounds channel")


if __name__ == "__main__":
    test_occlude_named_groups()
    print("occlude_named_groups OK")
    test_occlude_zero_coords()
    print("occlude_zero_coords OK")
    test_occlude_custom_dict()
    print("occlude_custom_dict OK")
    test_random_occlude_joint_groups()
    print("random_occlude_joint_groups OK")
    test_random_occlude_per_sample()
    print("random_occlude_per_sample OK")
    test_augmenter_shape_preserve()
    print("augmenter_shape_preserve OK")
    test_augmenter_temporal_consistency()
    print("augmenter_temporal_consistency OK")
    test_augmenter_mpiinf3dhp()
    print("augmenter_mpiinf3dhp OK")
    test_state_dict_round_trip()
    print("state_dict_round_trip OK")
    test_no_group_rate_returns_input_unchanged()
    print("no_group_rate_returns_input_unchanged OK")
    test_unsupported_skeleton_raises()
    print("unsupported_skeleton_raises OK")
    test_confidence_channel_validation()
    print("confidence_channel_validation OK")
    print("All synthetic occlusion augmentation smoke tests passed.")
