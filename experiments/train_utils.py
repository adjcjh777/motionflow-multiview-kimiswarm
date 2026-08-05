"""Training utilities for ray-aware attention fusion trainers.

Provides bone-length and skeleton-consistency losses that can be dropped into
the v3/v4 H36M trainers without changing model code.  The losses are fully
configurable via a parent-array or bone-pair list; a few common 17-joint
skeleton presets are included.

Summary (added during swarm-iter5 bone-length task):
- Added bone_length_loss(): supervised L1 on per-bone lengths vs. GT.
- Added temporal_bone_length_consistency_loss(): penalises bone-length std across batch.
- Added bone_symmetry_loss(): matches mirrored left/right bone lengths.
- Added skeleton_consistency_loss(): temporal + symmetry wrapper.
- Wired losses into experiments/train_ray_attention_v3_h36m.py with CLI args
  --bone_weight, --consistency_weight, --skeleton_layout.
- Verified imports and loss computation with random 17-joint tensors.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Skeleton presets (joint -> parent joint, -1 for root)
# ---------------------------------------------------------------------------

# H36M 17-joint subset used by many multi-view pose benchmarks.
#   0: pelvis          1: right_hip      2: right_knee     3: right_ankle
#   4: left_hip         5: left_knee       6: left_ankle     7: spine
#   8: neck             9: head           10: left_shoulder 11: left_elbow
#  12: left_wrist      13: right_shoulder 14: right_elbow   15: right_wrist
#  16: head_top
H36M_17_PARENTS = [
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14, 9,
]

# COCO 17 keypoint layout.
#   0: nose            1: left_eye       2: right_eye      3: left_ear
#   4: right_ear       5: left_shoulder   6: right_shoulder  7: left_elbow
#   8: right_elbow     9: left_wrist     10: right_wrist   11: left_hip
#  12: right_hip      13: left_knee      14: right_knee    15: left_ankle
#  16: right_ankle
COCO_17_PARENTS = [
    -1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14,
]

# SMPL first-17 joint layout (output.joints[..., :17]).
#   0: pelvis          1: left_hip        2: right_hip      3: spine1
#   4: left_knee       5: right_knee       6: spine2         7: left_ankle
#   8: right_ankle     9: spine3          10: left_foot      11: right_foot
#  12: neck           13: left_collar     14: right_collar   15: head
#  16: left_shoulder
SMPL17_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13,
]


# ---------------------------------------------------------------------------
# Bone helpers
# ---------------------------------------------------------------------------

def _parents_to_bone_pairs(parents: Sequence[int]) -> List[List[int]]:
    """Convert a parent array to a list of (parent, child) bone pairs."""
    return [[p, i] for i, p in enumerate(parents) if p >= 0]


def compute_bone_lengths(
    joints: torch.Tensor,
    parents: Optional[Sequence[int]] = None,
    bone_pairs: Optional[Sequence[Sequence[int]]] = None,
) -> torch.Tensor:
    """Compute per-bone lengths for a batch of 3D skeletons.

    Args:
        joints: (B, J, 3) predicted or target 3D joints.
        parents: Optional (J,) parent array; ignored if bone_pairs is given.
        bone_pairs: Optional list of [parent, child] joint indices.

    Returns:
        (B, N_bones) Euclidean bone lengths.
    """
    if bone_pairs is None:
        if parents is None:
            raise ValueError("Either parents or bone_pairs must be provided")
        bone_pairs = _parents_to_bone_pairs(parents)
    bone_pairs = torch.as_tensor(bone_pairs, dtype=torch.long, device=joints.device)
    bones = joints[:, bone_pairs]  # (B, N_bones, 2, 3)
    deltas = bones[:, :, 1, :] - bones[:, :, 0, :]  # (B, N_bones, 3)
    return deltas.norm(dim=-1)  # (B, N_bones)


def bone_length_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    parents: Optional[Sequence[int]] = None,
    bone_pairs: Optional[Sequence[Sequence[int]]] = None,
    weight: float = 1.0,
) -> torch.Tensor:
    """Supervised L1 loss on bone lengths.

    Args:
        pred:   (B, J, 3) predicted 3D pose.
        target: (B, J, 3) ground-truth 3D pose.
        parents / bone_pairs: Skeleton topology (see compute_bone_lengths).
        weight: Scalar multiplier.

    Returns:
        Scalar loss.
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=pred.device)
    pred_lengths = compute_bone_lengths(pred, parents=parents, bone_pairs=bone_pairs)
    target_lengths = compute_bone_lengths(target, parents=parents, bone_pairs=bone_pairs)
    return weight * F.l1_loss(pred_lengths, target_lengths)


# ---------------------------------------------------------------------------
# Skeleton consistency losses
# ---------------------------------------------------------------------------

def temporal_bone_length_consistency_loss(
    pred: torch.Tensor,
    parents: Optional[Sequence[int]] = None,
    bone_pairs: Optional[Sequence[Sequence[int]]] = None,
    weight: float = 1.0,
) -> torch.Tensor:
    """Temporal consistency of bone lengths across a batch/sequence.

    Penalizes the standard deviation of each bone length over the batch,
    encouraging constant bone lengths for a given subject.  Useful when the
    batch contains consecutive frames from one sequence; the loss still
    provides a weak regularizer for shuffled mini-batches.

    Args:
        pred:   (B, J, 3) predicted 3D pose.
        parents / bone_pairs: Skeleton topology.
        weight: Scalar multiplier.
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=pred.device)
    lengths = compute_bone_lengths(pred, parents=parents, bone_pairs=bone_pairs)
    return weight * lengths.std(dim=0).mean()


def bone_symmetry_loss(
    pred: torch.Tensor,
    left_bones: Sequence[int],
    right_bones: Sequence[int],
    weight: float = 1.0,
    parents: Optional[Sequence[int]] = None,
    bone_pairs: Optional[Sequence[Sequence[int]]] = None,
) -> torch.Tensor:
    """Penalize differences in length between mirrored left/right bones.

    Args:
        pred:   (B, J, 3) predicted 3D pose.
        left_bones:  Indices into the bone list for left-side bones.
        right_bones: Corresponding indices for right-side bones.
        weight: Scalar multiplier.
        parents / bone_pairs: Skeleton topology.
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=pred.device)
    lengths = compute_bone_lengths(pred, parents=parents, bone_pairs=bone_pairs)
    left = lengths[:, left_bones]
    right = lengths[:, right_bones]
    return weight * F.l1_loss(left, right)


def skeleton_consistency_loss(
    pred: torch.Tensor,
    parents: Optional[Sequence[int]] = None,
    bone_pairs: Optional[Sequence[Sequence[int]]] = None,
    left_bones: Optional[Sequence[int]] = None,
    right_bones: Optional[Sequence[int]] = None,
    temporal_weight: float = 1.0,
    symmetry_weight: float = 1.0,
) -> torch.Tensor:
    """Combined skeleton consistency loss (temporal + symmetry).

    Args:
        pred:   (B, J, 3) predicted 3D pose.
        parents / bone_pairs: Skeleton topology.
        left_bones / right_bones: Optional mirrored bone index lists.
        temporal_weight / symmetry_weight: Component multipliers.
    """
    loss = temporal_bone_length_consistency_loss(
        pred, parents=parents, bone_pairs=bone_pairs, weight=temporal_weight
    )
    if left_bones is not None and right_bones is not None:
        loss = loss + bone_symmetry_loss(
            pred, left_bones, right_bones, weight=symmetry_weight,
            parents=parents, bone_pairs=bone_pairs,
        )
    return loss
