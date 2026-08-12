# Mixed-dataset manifest validation: `configs/splits/mix_true_gt_v2.yaml`

> Validation date: 2026-08-11  
> Validator: coder subagent  
> GPU status at check: **busy** (RTX 4090 running v57 H36M true-GT medium). Validation was CPU-only; no GPU training/eval was launched.

## Scope

Verify that the mixed-dataset training manifest `configs/splits/mix_true_gt_v2.yaml` loads correctly with the WebBridge mixed loader (`motionflow_mv.data.webbridge_mixed_dataset`).

Checks performed:

1. YAML parses and contains required top-level keys (`name`, `description`, `domain_balancing_weights`).
2. `train_paths` / `train_names`, `val_paths` / `val_names`, and `test_paths` / `test_names` have matching lengths and consistent domain labels.
3. Every referenced `.npz` file exists on disk.
4. Representative files from each domain contain the required canonical keys.
5. `WebBridgeCanonical17Dataset` can load a sample from each domain and returns the expected canonical shapes.

## Validation script

`scripts/validate_mix_true_gt_v2.py`

The script is intentionally light (one representative `.npz` per domain per split) so it can run on CPU while the GPU is occupied.

## Manifest summary

| Split | Files | h36m | aist | shelf | campus |
|---|---:|---:|---:|---:|---:|
| train | 853 | 5 | 846 | 1 | 1 |
| val | 510 | 2 | 506 | 1 | 1 |
| test | 57 | 1 | 56 | 0 | 0 |

- All referenced files exist (0 missing).
- Path/name lengths match for all splits.
- Domain labels inferred from filenames match the manifest's `*_names` entries.

## Domain balancing weights

The manifest records inverse-frequency domain weights:

```yaml
h36m: 42.65
aist: 0.252069
shelf: 213.25
campus: 213.25
```

## `.npz` key checks

All required canonical keys (`points_2d`, `confidences`, `joints_3d`, `camera_K`, `camera_R`, `camera_t`) are present in the representative files checked for each split/domain.

## WebBridge loader checks

`WebBridgeCanonical17Dataset` successfully loaded a sample from every representative file:

| Split | Domain | Dataset length (clips) | Sample shapes (x, y, K, R, t, id) |
|---|---:|---:|---|
| train | h36m | 62,086 | `(9, 14, 17, 3)`, `(9, 17, 3)`, `(14, 3, 3)`, `(14, 3, 3)`, `(14, 3)`, `0` |
| train | aist | 712 | `(9, 14, 17, 3)`, `(9, 17, 3)`, `(14, 3, 3)`, `(14, 3, 3)`, `(14, 3)`, `2` |
| train | shelf | 338 | `(9, 14, 17, 3)`, `(9, 17, 3)`, `(14, 3, 3)`, `(14, 3, 3)`, `(14, 3)`, `3` |
| train | campus | 644 | `(9, 14, 17, 3)`, `(9, 17, 3)`, `(14, 3, 3)`, `(14, 3, 3)`, `(14, 3)`, `4` |
| val | h36m | 83,751 | same canonical shapes, id `0` |
| val | aist | 712 | same canonical shapes, id `2` |
| val | shelf | 79 | same canonical shapes, id `3` |
| val | campus | 156 | same canonical shapes, id `4` |
| test | h36m | 57,963 | same canonical shapes, id `0` |
| test | aist | 712 | same canonical shapes, id `2` |

All samples have the expected canonical layout:

- `x`: `(T, MAX_VIEWS, 17, 3)` — 2D keypoints + confidence
- `y`: `(T, 17, 3)` — 3D ground truth
- `K`, `R`: `(MAX_VIEWS, 3, 3)` — intrinsic / rotation
- `t`: `(MAX_VIEWS, 3)` — translation
- `dataset_id`: integer domain label (`0` h36m, `2` aist, `3` shelf, `4` campus)

`MAX_VIEWS = 14` and the 17-joint canonical skeleton are correctly padded for all sources, including the 4-view H36M/Shelf/Campus and 9-view AIST++ files.

## Observations

1. **Test split has no Shelf/Campus hold-out.** Only H36M and AIST++ files appear in `test_paths`. This is consistent with the manifest's comment that a single-file val pool is kept in val so val stays non-empty; however, it means the mixed-dataset test set does not exercise Shelf/Campus generalisation.
2. **Shelf/Campus representation is small.** Train has one Shelf and one Campus file; val has one each. The inverse-frequency weights for shelf/campus (`213.25`) are very high relative to their counts.
3. **GPU was not used.** The entire validation ran on CPU and completed without issue.

## Conclusion

`configs/splits/mix_true_gt_v2.yaml` is **valid and loadable** by the WebBridge mixed loader:

- YAML structure is correct.
- All 1,420 referenced `.npz` files exist.
- Path/name alignment and domain labels are consistent.
- Representative files from every domain produce the expected canonical sample shapes.
- No GPU resources were consumed.

No blockers for using this manifest in CPU-only preparation or, once the GPU is free, for launching training/eval runs that consume it.
