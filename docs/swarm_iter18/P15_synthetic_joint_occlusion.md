# P15 Synthetic Joint Occlusion Augmentation

**Branch:** `feat/swarm-iter18-omniview`  
**Author:** Kimi Code subagent  
**Date:** 2026-08-07  
**Status:** Implemented + smoke-tested

## 1. Goal

Add a reusable, deterministic synthetic occlusion augmenter that drops *groups* of anatomically related joints (limbs, head, torso) instead of independent random joints. This matches real occlusions more closely and provides supervision for the visibility head in OmniMultiViewFusion v2.

## 2. Files

- `motionflow_mv/data/synthetic_occlusion_aug.py` — main module.
  - `occlude_joint_groups(...)` — deterministic group occlusion.
  - `random_occlude_joint_groups(...)` — stochastic group occlusion.
  - `SyntheticJointOcclusionAugmenter` — drop-in augmenter with group + per-joint occlusion, optional temporal consistency, and reproducible `state_dict`/`load_state_dict`.
- `tests/test_synthetic_occlusion_aug.py` — CPU smoke tests.

## 3. Supported skeletons

| Alias | Joints | Groups |
|-------|--------|--------|
| `h36m_17` | 17 | torso, head, left_arm, right_arm, left_leg, right_leg |
| `mpiinf3dhp_28` | 28 | head, torso, left_arm, right_arm, left_leg, right_leg |

Custom group dictionaries are also accepted.

## 4. Usage example

```python
from motionflow_mv.data.synthetic_occlusion_aug import SyntheticJointOcclusionAugmenter

augmenter = SyntheticJointOcclusionAugmenter(
    skeleton="h36m_17",
    group_rate=0.3,
    joint_rate=0.05,
    temporal_consistency=True,
    zero_coords=True,
    seed=42,
)
out = augmenter(x)  # x: (B, T, V, J, 3)
```

## 5. Smoke test result

```text
All synthetic occlusion augmentation smoke tests passed.
```

Run with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_synthetic_occlusion_aug.py
```

## 6. Next steps

1. Wire `SyntheticJointOcclusionAugmenter` into the WebBridge / MPI-INF-3DHP dataloader pipeline.
2. Use the generated occlusion masks as auxiliary labels for the visibility BCE loss in OmniMultiViewFusion v2.
3. Ablate `group_rate` vs. plain per-joint dropout on MPI-INF-3DHP S1 smoke runs.
