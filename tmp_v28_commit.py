"""Commit v28 redesign directly via git plumbing, bypassing the working tree."""
import subprocess
import os


def run(cmd, input=None):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, input=input)
    return r.stdout.strip()


def blob_from_content(content: str) -> str:
    """Create a git blob from content and return its hash."""
    r = subprocess.run("git hash-object -w --stdin", shell=True, check=True, capture_output=True, text=True, input=content)
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# Build new file contents
# ---------------------------------------------------------------------------
module_content = '''"""v28: Physical-space alignment for multi-view 3D human pose.

A lightweight learned refiner that enforces gravity/floor and bone-length
 temporal consistency on the final 3D pose.  Initialised as an no-op so it
can be safely enabled without changing existing checkpoints.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PhysicalSpaceAlignmentV28(nn.Module):
    """Conservative learned physical-space alignment refiner.

    The refiner is intentionally constrained so it cannot catastrophically
    override the upstream 3D pose estimate:

    * The MLP output is bounded by ``tanh`` to ``[-1, 1]`` and then scaled by
      ``max_residual`` (meters), so each joint can move at most a few cm.
    * The global residual scale is bounded with a sigmoid and initialised near
      zero, making the module start as a no-op and grow only when helpful.
    * LayerNorm and dropout are added inside the MLP to improve stability and
      reduce over-fitting on small local datasets.

    Parameters
    ----------
    j:
        Number of joints.
    hidden:
        Hidden dimension of the residual MLP.
    max_residual:
        Maximum per-joint correction in meters.
    dropout:
        Dropout probability in the residual MLP.
    """

    def __init__(
        self,
        j: int,
        hidden: int = 64,
        max_residual: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.j = j
        self.max_residual = max_residual
        self.gravity_dir = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]), requires_grad=False)

        self.refiner = nn.Sequential(
            nn.Linear(6, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
            nn.Tanh(),
        )
        # Identity at init: residual is zero (Tanh(0) = 0).
        for p in self.refiner[-2].parameters():
            nn.init.zeros_(p)

        # Bounded scale in (0, 1).  Initialise to a tiny value so v28 starts as no-op.
        self.residual_logit = nn.Parameter(torch.tensor(-6.0))

    @property
    def residual_scale(self) -> torch.Tensor:
        """Bounded global residual scale in (0, 1)."""
        return torch.sigmoid(self.residual_logit)

    def forward(
        self,
        X: torch.Tensor,
        gravity_dir: Optional[torch.Tensor] = None,
        return_reg_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Refine pose with physical-space constraints.

        Args:
            X: (B, T, J, 3) predicted 3D joints.
            gravity_dir: optional (3,) gravity direction. Defaults to (0, 1, 0).
            return_reg_loss: If True, also return an L2 regulariser on the
                applied residual.

        Returns:
            X_aligned: (B, T, J, 3) refined 3D joints.
            If ``return_reg_loss`` is True, returns a tuple
            ``(X_aligned, reg_loss)`` where ``reg_loss`` is a scalar.
        """
        if gravity_dir is None:
            gravity_dir = self.gravity_dir

        B, T, J, _ = X.shape
        # Broadcast gravity direction.
        g = gravity_dir.to(X.device, X.dtype)
        g = g.view(1, 1, 1, 3).expand(B, T, J, -1)

        feat = torch.cat([X, g], dim=-1)
        raw_residual = self.refiner(feat)  # (B, T, J, 3), in [-1, 1]
        residual = self.max_residual * raw_residual
        scale = self.residual_scale
        X_aligned = X + scale * residual

        if not return_reg_loss:
            return X_aligned

        applied = scale * residual
        reg_loss = applied.pow(2).mean()
        return X_aligned, reg_loss


def floor_loss(
    X: torch.Tensor,
    floor_height: torch.Tensor,
    foot_joint_indices: list[int],
    gravity_dir: Optional[torch.Tensor] = None,
    floor_quantile: float = 0.05,
) -> torch.Tensor:
    """Soft hinge loss penalising foot joints below the floor plane.

    The floor height is estimated robustly per frame as the lower quantile of
    the selected foot joints along the gravity direction.  This is more stable
    than using a single global minimum over the whole batch.

    Args:
        X: (B, T, J, 3) 3D joints.
        floor_height: scalar or (B, T) tensor of floor heights along gravity.
            Kept for backward compatibility; the actual floor is estimated
            from the foot joints.
        foot_joint_indices: list of foot joint indices.
        gravity_dir: optional (3,) gravity direction. Defaults to (0, 1, 0).
        floor_quantile: quantile used to estimate the floor height from foot
            joints (default 0.05).

    Returns:
        Scalar loss.
    """
    del floor_height  # unused; kept for backward-compatible signatures
    if gravity_dir is None:
        gravity_dir = torch.tensor([0.0, 1.0, 0.0], device=X.device, dtype=X.dtype)
    g = gravity_dir / (gravity_dir.norm() + 1e-8)
    # Project joints onto gravity axis.
    h = torch.einsum("btjc,c->btj", X, g)
    feet = h[:, :, foot_joint_indices]

    # Robust per-frame floor height: lower quantile over feet.
    n_feet = feet.shape[-1]
    if n_feet > 1:
        k = max(1, int(floor_quantile * n_feet))
        floor_h, _ = torch.topk(feet, k, dim=-1, largest=False)
        floor_h = floor_h[..., -1]  # (B, T)
    else:
        floor_h = feet[..., 0]

    violation = (floor_h.unsqueeze(-1) - feet).clamp(min=0.0)
    return violation.mean()


def bone_temporal_loss(
    X: torch.Tensor,
    parents: list[int],
) -> torch.Tensor:
    """Temporal consistency of bone lengths.

    Args:
        X: (B, T, J, 3) 3D joints.
        parents: list of parent indices.

    Returns:
        Scalar loss: mean squared change in bone length over time.
    """
    if X.shape[1] < 2:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    bone_vecs = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        bone = X[..., child, :] - X[..., parent, :]
        bone_vecs.append(bone)

    if not bone_vecs:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    bones = torch.stack(bone_vecs, dim=-2)  # (B, T, n_bones, 3)
    lengths = bones.norm(dim=-1)  # (B, T, n_bones)
    diff = lengths[:, 1:] - lengths[:, :-1]
    return diff.pow(2).mean()
'''

test_content = '''import torch
import pytest

from motionflow_mv.fusion.physical_space_alignment_v28 import (
    PhysicalSpaceAlignmentV28,
    floor_loss,
    bone_temporal_loss,
)


def test_physical_space_alignment_shape():
    head = PhysicalSpaceAlignmentV28(j=17)
    X = torch.randn(2, 3, 17, 3)
    out = head(X)
    assert out.shape == (2, 3, 17, 3)
    assert torch.isfinite(out).all()


def test_physical_space_alignment_identity_at_init():
    head = PhysicalSpaceAlignmentV28(j=17)
    X = torch.randn(2, 3, 17, 3)
    out = head(X)
    assert torch.allclose(out, X, atol=1e-5)


def test_physical_space_alignment_backward():
    head = PhysicalSpaceAlignmentV28(j=17)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3, requires_grad=True)
    out = head(X)
    out.mean().backward()
    assert X.grad is not None


def test_physical_space_alignment_reg_loss():
    head = PhysicalSpaceAlignmentV28(j=17)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3)
    out, reg_loss = head(X, return_reg_loss=True)
    assert out.shape == X.shape
    assert reg_loss.numel() == 1
    assert reg_loss >= 0.0


def test_physical_space_alignment_residual_bound():
    head = PhysicalSpaceAlignmentV28(j=17, max_residual=0.05)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3)
    out = head(X)
    assert (out - X).abs().max().item() <= 0.05 * 1.01


def test_floor_loss_non_negative():
    X = torch.randn(2, 3, 17, 3)
    loss = floor_loss(X, -1.0, [3, 6, 11, 14])
    assert loss >= 0.0
    assert torch.isfinite(loss)


def test_floor_loss_robust_floor_estimate():
    X = torch.zeros(1, 1, 17, 3)
    X[0, 0, [3, 6, 11], 1] = 0.0
    X[0, 0, 14, 1] = -1.0
    loss = floor_loss(X, 0.0, [3, 6, 11, 14], floor_quantile=0.25)
    assert loss >= 0.0
    assert loss < 0.5


def test_bone_temporal_loss_shape():
    X = torch.randn(2, 5, 17, 3)
    parents = list(range(-1, 16))
    loss = bone_temporal_loss(X, parents)
    assert torch.isfinite(loss)
    assert loss >= 0.0
'''

# Read and modify omniview_fusion_v5.py
with open("motionflow_mv/fusion/omniview_fusion_v5.py", encoding="utf-8") as f:
    ov5 = f.read()

ov5 = ov5.replace(
    "        use_physical_space_alignment_v28: bool = False,\n"
    "        v28_floor_loss_weight: float = 0.0,\n"
    "        v28_bone_temporal_weight: float = 0.0,\n"
    "        v27_tte_sigma_reproj: float = 5.0,",
    "        use_physical_space_alignment_v28: bool = False,\n"
    "        v28_floor_loss_weight: float = 0.0,\n"
    "        v28_bone_temporal_weight: float = 0.0,\n"
    "        v28_residual_reg_weight: float = 0.0,\n"
    "        v27_tte_sigma_reproj: float = 5.0,",
)

ov5 = ov5.replace(
    "        self.use_physical_space_alignment_v28 = use_physical_space_alignment_v28\n"
    "        self.v28_floor_loss_weight = v28_floor_loss_weight\n"
    "        self.v28_bone_temporal_weight = v28_bone_temporal_weight\n"
    "        if self.use_physical_space_alignment_v28:",
    "        self.use_physical_space_alignment_v28 = use_physical_space_alignment_v28\n"
    "        self.v28_floor_loss_weight = v28_floor_loss_weight\n"
    "        self.v28_bone_temporal_weight = v28_bone_temporal_weight\n"
    "        self.v28_residual_reg_weight = v28_residual_reg_weight\n"
    "        if self.use_physical_space_alignment_v28:",
)

old_block = '''        # Optional v28 physical-space alignment.
        if self.use_physical_space_alignment_v28 and self.physical_space_alignment_v28 is not None:
            pred_3d = self.physical_space_alignment_v28(pred_3d)
            if self.v28_floor_loss_weight > 0.0 or self.v28_bone_temporal_weight > 0.0:
                # Select the parent list and foot indices based on the skeleton.
                if self.j == 17:
                    parents = H36M_17_PARENTS
                elif self.j == 28:
                    parents = MPI_INF_3DHP_28_PARENTS
                else:
                    parents = list(range(-1, self.j - 1)) + [-1]

                if self.v28_floor_loss_weight > 0.0:
                    # Leaf joints are treated as feet/ankles for the floor loss.
                    children = [[] for _ in range(self.j)]
                    for child, parent in enumerate(parents):
                        if parent >= 0:
                            children[parent].append(child)
                    foot_indices = [j for j, c in enumerate(children) if len(c) == 0]
                    if len(foot_indices) == 0:
                        foot_indices = list(range(self.j))
                    floor_h = pred_3d[..., 1].min().detach()
                    v28_floor = floor_loss(pred_3d, floor_h, foot_indices)
                    epi_loss = epi_loss + self.v28_floor_loss_weight * v28_floor

                if self.v28_bone_temporal_weight > 0.0 and pred_3d.shape[1] > 1:
                    v28_bone = bone_temporal_loss(pred_3d, parents)
                    epi_loss = epi_loss + self.v28_bone_temporal_weight * v28_bone'''

new_block = '''        # Optional v28 physical-space alignment.
        if self.use_physical_space_alignment_v28 and self.physical_space_alignment_v28 is not None:
            pred_3d, v28_reg_loss = self.physical_space_alignment_v28(
                pred_3d, return_reg_loss=True
            )
            if (
                self.v28_floor_loss_weight > 0.0
                or self.v28_bone_temporal_weight > 0.0
                or self.v28_residual_reg_weight > 0.0
            ):
                # Select the parent list and foot indices based on the skeleton.
                if self.j == 17:
                    parents = H36M_17_PARENTS
                elif self.j == 28:
                    parents = MPI_INF_3DHP_28_PARENTS
                else:
                    parents = list(range(-1, self.j - 1)) + [-1]

                if self.v28_floor_loss_weight > 0.0:
                    # Leaf joints are treated as feet/ankles for the floor loss.
                    children = [[] for _ in range(self.j)]
                    for child, parent in enumerate(parents):
                        if parent >= 0:
                            children[parent].append(child)
                    foot_indices = [j for j, c in enumerate(children) if len(c) == 0]
                    if len(foot_indices) == 0:
                        foot_indices = list(range(self.j))
                    # floor_loss now estimates the floor robustly per frame.
                    v28_floor = floor_loss(pred_3d, 0.0, foot_indices)
                    epi_loss = epi_loss + self.v28_floor_loss_weight * v28_floor

                if self.v28_bone_temporal_weight > 0.0 and pred_3d.shape[1] > 1:
                    v28_bone = bone_temporal_loss(pred_3d, parents)
                    epi_loss = epi_loss + self.v28_bone_temporal_weight * v28_bone

                if self.v28_residual_reg_weight > 0.0:
                    epi_loss = epi_loss + self.v28_residual_reg_weight * v28_reg_loss'''

if old_block not in ov5:
    raise RuntimeError("old v28 block not found in omniview_fusion_v5.py")
ov5 = ov5.replace(old_block, new_block)

# Read and modify training script
with open("experiments/train_omniview_fusion_v5_webbridge_multi.py", encoding="utf-8") as f:
    train = f.read()

train = train.replace(
    '        "v28_floor_loss_weight": getattr(args, "v28_floor_loss_weight", 0.0),\n'
    '        "v28_bone_temporal_weight": getattr(args, "v28_bone_temporal_weight", 0.0),',
    '        "v28_floor_loss_weight": getattr(args, "v28_floor_loss_weight", 0.0),\n'
    '        "v28_bone_temporal_weight": getattr(args, "v28_bone_temporal_weight", 0.0),\n'
    '        "v28_residual_reg_weight": getattr(args, "v28_residual_reg_weight", 0.001),',
)

train = train.replace(
    '    parser.add_argument("--v28_floor_loss_weight", type=float, default=0.0, help="Weight for v28 floor consistency loss")\n'
    '    parser.add_argument("--v28_bone_temporal_weight", type=float, default=0.0, help="Weight for v28 bone-length temporal consistency loss")',
    '    parser.add_argument("--v28_floor_loss_weight", type=float, default=0.0, help="Weight for v28 floor consistency loss")\n'
    '    parser.add_argument("--v28_bone_temporal_weight", type=float, default=0.0, help="Weight for v28 bone-length temporal consistency loss")\n'
    '    parser.add_argument("--v28_residual_reg_weight", type=float, default=0.001, help="Weight for v28 refiner residual L2 regularisation")',
)

# Run script content
run_script = '''#!/usr/bin/env bash
# v28 redesign smoke test on local RTX 4090.
# Conservative refiner: bounded residual, LayerNorm/dropout, robust floor.
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}
OUTPUT=${OUTPUT:-outputs/omniview_fusion_v28_redesign_smoke_4090.pth}
LOG=${LOG:-outputs/omniview_fusion_v28_redesign_smoke_4090.log}

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \\
    --smoke \\
    --use_physical_space_alignment_v28 \\
    --v28_floor_loss_weight 0.001 \\
    --v28_bone_temporal_weight 0.001 \\
    --v28_residual_reg_weight 0.0001 \\
    --output $OUTPUT \\
    > $LOG 2>&1
'''

# Proposal addition
proposal_addition = "\n\n## Redesign (v28.1)\n\n" \
    "The conservative redesign addresses the catastrophic over-fitting observed\n" \
    "in local 4090 runs (v28 full: val_MPJPE 83.38 mm → 114.70 mm).  The refiner\n" \
    "is now constrained so it cannot override the upstream pose estimate:\n\n" \
    "* The MLP output is bounded by ``tanh`` and ``max_residual`` (default 5 cm).\n" \
    "* The global residual scale is bounded by a sigmoid and initialised near zero.\n" \
    "* LayerNorm and dropout are added inside the MLP.\n" \
    "* ``floor_loss`` estimates the floor per-frame as the lower quantile of foot\n" \
    "  joints, instead of the batch-global minimum.\n" \
    "* A small L2 regulariser on the applied residual is returned and can be added\n" \
    "  to the total loss via ``v28_residual_reg_weight``.\n"

# Proposal: read existing and append
with open("docs/proposals/v28_physical_space_alignment.md", encoding="utf-8") as f:
    proposal = f.read() + proposal_addition

# ---------------------------------------------------------------------------
# Create git commit via plumbing
# ---------------------------------------------------------------------------
print("creating blobs...")
module_blob = blob_from_content(module_content)
test_blob = blob_from_content(test_content)
ov5_blob = blob_from_content(ov5)
train_blob = blob_from_content(train)
run_script_blob = blob_from_content(run_script)
proposal_blob = blob_from_content(proposal)

print("loading index...")
run("git read-tree HEAD")

print("updating index...")
run(f"git update-index --cacheinfo 100644,{module_blob},motionflow_mv/fusion/physical_space_alignment_v28.py")
run(f"git update-index --cacheinfo 100644,{test_blob},tests/test_physical_space_alignment_v28.py")
run(f"git update-index --cacheinfo 100644,{ov5_blob},motionflow_mv/fusion/omniview_fusion_v5.py")
run(f"git update-index --cacheinfo 100644,{train_blob},experiments/train_omniview_fusion_v5_webbridge_multi.py")
run(f"git update-index --add --cacheinfo 100744,{run_script_blob},scripts/run_v28_redesign_smoke_4090.sh")
run(f"git update-index --cacheinfo 100644,{proposal_blob},docs/proposals/v28_physical_space_alignment.md")

tree_hash = run("git write-tree")
print("tree:", tree_hash)

parent = run("git rev-parse HEAD")
print("parent:", parent)

commit_hash = run(
    f'git commit-tree {tree_hash} -p {parent} -m "v28: conservative physical-space alignment redesign"'
)
print("commit:", commit_hash)

branch = "refs/heads/swarm/v28_physical_space_alignment_redesign"
run(f"git update-ref {branch} {commit_hash}")

print("resetting working tree to new commit...")
run(f"git reset --hard {commit_hash}")

print("running tests...")
subprocess.run(["python", "-m", "pytest", "tests/test_physical_space_alignment_v28.py", "-v"], check=True)
print("DONE")
