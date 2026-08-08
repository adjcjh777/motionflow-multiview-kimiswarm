# v28: Physical-Space Alignment for Multi-View Pose Estimation

## Motivation

Current motionflow predicts 3D human poses in an implicit camera-centric
coordinate system.  For multi-view video captured in a real physical space,
the same action should be aligned to a shared world frame regardless of which
cameras are used.  v28 adds a lightweight **physical-space alignment (PSA)**
module that enforces:

1. **Gravity/floor consistency**: the estimated skeleton should share a
   consistent upright direction and foot-floor contact across the sequence.
2. **Bone-length temporal consistency**: bone lengths should vary slowly over
   time, reflecting physical rigidity.
3. **Camera-relative plausibility**: the estimated 3D joints should project back
   to all visible views with low reprojection error (already partly covered by
   v25/v26, but PSA makes it an explicit physical constraint).

## Design

The module is inserted after the final 3D pose prediction and refines the
pose by solving a small optimization problem with a neural prior:

```
min  w_data * ||X - X_in||^2 + w_floor * L_floor(X) + w_bone * L_bone(X)
```

where:
- `X_in` is the pose produced by the upstream network.
- `L_floor(X)` penalises downward foot joints below the estimated floor plane.
- `L_bone(X)` penalises large frame-to-frame changes in bone lengths.

The optimization is replaced by a learned MLP refiner to keep training
simple and fast:

```
X_aligned = X_in + residual_scale * MLP([X_in, gravity_dir, floor_height])
```

The residual MLP is initialised to zero, so v28 starts as an no-op.

## Interface

```python
class PhysicalSpaceAlignmentV28(nn.Module):
    def __init__(self, j: int, hidden: int = 64):
        ...

    def forward(self, X: torch.Tensor, gravity_dir: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args: X (B, T, J, 3). Returns: X_aligned (B, T, J, 3)."""
```

## Losses

- `floor_loss`: soft hinge on foot joints below the floor plane.
- `bone_temporal_loss`: MSE of bone lengths between consecutive frames.

These are added to the total training loss when v28 is enabled.

## Training

Added as a flag in `OmniMultiViewFusionV5`:
- `use_physical_space_alignment_v28`
- `v28_floor_loss_weight`
- `v28_bone_temporal_weight`

## Expected Impact

- Better cross-view consistency when cameras are at very different heights.
- More physically plausible 3D trajectories over time.
- Improved robustness to calibration errors that affect the world-frame
  interpretation of the skeleton.
