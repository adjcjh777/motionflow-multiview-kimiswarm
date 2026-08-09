# v40 Physical Loss Improvements

## Current state

`motionflow_mv/losses/skeleton_physical_loss_v40.py` implements a composite
skeleton-aware physical loss with four terms:

1. Bone-length matching against ground truth.
2. Soft joint-angle (hyper-extension) limit.
3. Left/right symmetry of bone lengths.
4. Foot-floor contact / penetration prior.

While this already helps constrain poses, two important physical cues are either
missing or modeled too crudely. This document proposes concrete improvements
that can be implemented without changing the existing training loop API.

---

## Proposal 1: Add a self-collision / self-penetration penalty

### Problem

Bone-length and symmetry only preserve the shape of individual limbs. They do
not prevent anatomically impossible configurations such as an arm intersecting
the torso or the two legs crossing each other inside the hip.

### Solution

Reuse the existing `PhysicalCollisionPenaltyV31` from
`motionflow_mv/losses/physical_collision_penalty_v31.py` as a new term inside
`SkeletonPhysicalLossV40`.

### Concrete implementation

- Add to `SkeletonPhysicalLossV40.__init__`:
  - `collision_weight: float = 0.001`
  - `collision_margin: float = 0.05`
- Instantiate once:

```python
from motionflow_mv.losses.physical_collision_penalty_v31 import PhysicalCollisionPenaltyV31

self.collision_loss = PhysicalCollisionPenaltyV31(
    parents=parents,
    loss_weight=1.0,
    margin=collision_margin,
    warmup_epochs=0,
)
```

- In `forward`, if `collision_weight > 0.0`:

```python
# (B, T, J, 3) -> (B*T, J, 3)
bt_pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
collision_loss, _ = self.collision_loss(bt_pred)
loss = loss + ramp * collision_weight * collision_loss
```

- Expose CLI args in `experiments/train_omniview_fusion_v5_webbridge_multi.py`:
  - `--v40_collision_weight`
  - `--v40_collision_margin`

### Expected impact

Reduces inter-penetration of non-adjacent bones, making predicted poses more
anatomically plausible. The heavy lifting (capsule-capsule distance) already
exists in the repo, so the change is small.

---

## Proposal 2: Replace the naive floor prior with a contact-aware, data-driven floor model

### Problem

The current `floor_loss` assumes the floor is exactly at `y = 0` and penalizes
foot velocity equally across all frames:

```python
penetration = torch.relu(-feet[..., 1]).mean()
velocity = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1).mean()
return penetration + 0.1 * velocity
```

This is problematic because:

- Root-relative 3-D poses often do not have the floor at `y = 0`.
- Penalizing foot velocity even when a foot is clearly in the air suppresses
  natural motion (e.g., walking, jumping).

### Solution

Estimate the floor height robustly from the pose itself (or allow an external
override), and only penalize foot velocity when the foot is actually near the
floor.

### Concrete implementation

- Extend `SkeletonPhysicalLossV40.__init__` with:
  - `floor_height: Optional[float] = None` (`None` = auto-estimate)
  - `contact_threshold: float = 0.05`
  - `contact_velocity_weight: float = 0.1`

- Add a helper:

```python
def _estimate_floor_height(self, pred: torch.Tensor) -> torch.Tensor:
    """Robust floor estimate from lower-body joints.

    pred: (B, T, J, 3) or (B, J, 3)
    """
    y = pred[..., 1]  # all joints y-coordinate
    # Use 5th percentile to ignore swinging feet in the air.
    return torch.quantile(y.flatten(), 0.05)
```

- Rewrite `floor_loss`:

```python
def floor_loss(self, pred: torch.Tensor) -> torch.Tensor:
    if not self.foot_indices:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    feet = pred[..., self.foot_indices, :]  # (B, T, n_feet, 3)
    floor_h = self.floor_height
    if floor_h is None:
        floor_h = self._estimate_floor_height(pred)

    # Penetrate below the floor
    penetration = torch.relu(-(feet[..., 1] - floor_h)).mean()

    if feet.dim() == 3:
        return penetration

    # Contact-aware velocity: only penalize fast feet when near the floor.
    foot_speed = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
    foot_y = feet[:, 1:, ..., 1]
    near_floor = (foot_y - floor_h).abs() < self.contact_threshold
    velocity = (foot_speed * near_floor.float()).mean()

    return penetration + self.contact_velocity_weight * velocity
```

- Expose CLI args:
  - `--v40_floor_height` (default `None` for auto)
  - `--v40_contact_threshold`
  - `--v40_contact_velocity_weight`

### Expected impact

Better handling of root-relative coordinate systems and more realistic foot
contact behavior. The model will be allowed to move feet freely while in the
air, but still encouraged to keep planted feet steady and prevent floor
penetration.

---

## Integration notes

- Both new terms should go through the existing `ramp = self._ramp(epoch)`
  warm-up.
- Collision and floor terms should be optional and default-off to preserve
  backward compatibility with existing v40 configs.
- No GPU training should be launched until these proposals are reviewed and
  smoke-tested.
