# 3DPW Multi-View Conversion (swarm_iter5)

## Summary

Downloaded the 3DPW archive and converted its single-camera + IMU sequences into the project's canonical multi-view ``.npz`` format.  A new script
`experiments/convert_3dpw_multiview.py` produces either:

* **pseudo** multi-view: a static virtual camera rig placed around the actor, with the 24 SMPL joints re-projected into each view.
* **actual** single-view: the real moving camera stored as a single view, with per-frame intrinsics/extrinsics preserved in extra arrays.

The converted data loads directly into the existing `RayAttentionFusionModelTemporal` training/evaluation pipeline.

## Files

* `experiments/convert_3dpw_multiview.py` — converter (single-file and batch modes).
* `data/webbridge/3dpw/converted/{train,validation,test}/` — 56 converted sequences (4-view pseudo rig, person 0).

## How to run

Single sequence:

```bash
conda run -n mf python experiments/convert_3dpw_multiview.py \
    --input data/webbridge/3dpw/sequenceFiles/sequenceFiles/validation/courtyard_basketball_01.pkl \
    --output data/webbridge/3dpw/validation/courtyard_basketball_01_pseudo.npz \
    --mode pseudo --n_views 4 --noise_std 5.0
```

Batch (all splits):

```bash
for split in train validation test; do
    conda run -n mf python experiments/convert_3dpw_multiview.py \
        --input data/webbridge/3dpw/sequenceFiles/sequenceFiles/$split \
        --output data/webbridge/3dpw/converted/$split \
        --mode pseudo --n_views 4
done
```

## Verification

1. **Format check** — loaded `data/webbridge/3dpw/validation/courtyard_basketball_01_pseudo.npz` with the existing `TemporalClipDataset`; model forward pass succeeded with shapes `(B, T, 4, 24, 3)` -> `(B, T, 24, 3)`.

2. **Smoke training** (2 epochs, clip_len=13, batch_size=4) on two train sequences and one validation sequence:

| Data | val MPJPE |
|---|---|
| Pseudo noise-free | 0.00 mm |
| Pseudo + 5 px noise | 15.88 mm |

The 0 mm result on noise-free data is expected: the pseudo views are exact projections of the ground-truth 3D joints, so DLT triangulation recovers the joints almost perfectly.  Adding `--noise_std 5.0` makes the validation more realistic.

## Important findings / caveats

* 3DPW is **not a real multi-view dataset**.  The pseudo rig is synthetic; it is useful for sanity-checking the multi-view pipeline and for domain-transfer experiments, but it does not test true multi-view fusion on synchronized real cameras.
* Only the **first actor** (`person_idx=0`) is extracted by default.  Some 3DPW sequences contain two people; use `--person_idx 1` for the second actor.
* The `actual` mode stores the true per-frame camera in `camera_K_frames`, `camera_R_frames`, and `camera_t_frames`.  The canonical static `camera_K/R/t` slots hold the first frame as a placeholder, so existing code that expects static cameras runs unchanged.
* `imageFiles.zip` in the downloaded archive appears incomplete/corrupted in our copy; conversion does not require images because `jointPositions` already contains the 3D joints.

## Blockers

* None for the conversion itself.  Real in-the-wild multi-view evaluation remains blocked by the lack of an actual synchronized multi-view in-the-wild dataset.

## Dependencies

No new dependencies.  The script uses only NumPy and the project's `Camera` class.
