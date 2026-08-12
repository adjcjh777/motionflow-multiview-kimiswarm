# Bone-length and skeleton consistency loss

## What changed

- Added `experiments/train_utils.py` with reusable loss helpers:
  - `bone_length_loss(pred, target, parents/parent_pairs, weight)` — supervised L1 on bone lengths.
  - `temporal_bone_length_consistency_loss(...)` — penalises bone-length variance across the batch/sequence.
  - `bone_symmetry_loss(...)` — L1 difference between mirrored left/right bone lengths.
  - `skeleton_consistency_loss(...)` — combined temporal + symmetry wrapper.
  - Skeleton presets: `H36M_17_PARENTS`, `COCO_17_PARENTS`, `SMPL17_PARENTS`.
- Wired the losses into `experiments/train_ray_attention_v3_h36m.py`:
  - `--bone_weight` (default 0.1) enables supervised bone-length L1.
  - `--consistency_weight` (default 0.05) enables temporal + symmetry regulariser.
  - `--skeleton_layout` selects `h36m17` / `coco17` / `smpl17` topology.

## Verification

- Import check passed with `/d/anaconda3/envs/jz_py310/python.exe`.
- Loss sanity test with random 17-joint tensors produced finite scalar values for all loss modes and correctly returned 0 when `weight=0`.
- `python -m py_compile experiments/train_ray_attention_v3_h36m.py` succeeded.

## Notes / next steps

- The H36M parent/symmetry mapping should be double-checked against the actual joint order in `data/h36m_hf/*.npz`; the current preset follows the common 17-joint H36M subset.
- The bone losses are only applied during training; validation still reports pure MSE/MPJPE so the regularisation effect is isolated.
- A future v4 trainer can import the same `experiments/train_utils.py` helpers.
