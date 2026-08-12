# Residual Temporal Model on 3DPW Pseudo-Multi-View Data

## Goal

Run the current top-performing residual temporal ray-attention model
(`outputs/ray_attention_temporal_residual_v2.pth`) on the converted 3DPW
pseudo-multi-view data and report MPJPE, documenting any domain-gap issues.

## What was done

1. **Inspected the checkpoint and the converted 3DPW data.**
   - Checkpoint: `outputs/ray_attention_temporal_residual_v2.pth` (residual
     temporal model, class `RayAttentionFusionModelTemporalResidual`).
   - Converted 3DPW pseudo data (default): `data/webbridge/3dpw/converted/*`
     with 4 pseudo views and 24 SMPL joints per sequence.

2. **Discovered a view-count mismatch.**
   - The checkpoint contains `fusion_mlp.0.weight` of shape `(64, 896)`, i.e.
     `d * n_views = 64 * 14`, because it was trained on MPI-INF-3DHP with
     **14 cameras**.
   - The existing 4-view 3DPW conversion (`--n_views 4`) therefore cannot be
     evaluated directly; the model's `fusion_mlp` has a hard-coded 14-view
     input size.

3. **Re-converted 3DPW with 14 pseudo views** to match the checkpoint.

   ```bash
   for split in train validation test; do
       conda run -n mf python experiments/convert_3dpw_multiview.py \
           --input data/webbridge/3dpw/sequenceFiles/sequenceFiles/$split \
           --output data/webbridge/3dpw/converted_14views/$split \
           --mode pseudo --n_views 14
   done
   ```

4. **Created an evaluation script.**
   - `experiments/eval_residual_3dpw_pseudo.py`
   - Loads the residual temporal checkpoint (non-strict, because the checkpoint
     has a few extra / view-count-specific keys).
   - Evaluates non-overlapping 13-frame clips.
   - Reports both **raw** and **root-aligned** MPJPE in millimetres.

5. **Ran evaluation on the 14-view validation and test splits.**

   ```bash
   # validation
   conda run -n mf python experiments/eval_residual_3dpw_pseudo.py \
       --split data/webbridge/3dpw/converted_14views/validation \
       --clip_len 13 --batch_size 2 \
       --save_json outputs/residual_3dpw_pseudo_val.json

   # test
   conda run -n mf python experiments/eval_residual_3dpw_pseudo.py \
       --split data/webbridge/3dpw/converted_14views/test \
       --clip_len 13 --batch_size 2 \
       --save_json outputs/residual_3dpw_pseudo_test.json
   ```

## Results

| Split | Sequences | Mean raw MPJPE | Mean root-aligned MPJPE |
|-------|-----------|----------------|-------------------------|
| 3DPW validation (14-view pseudo) | 12 | **678.65 mm** | **81.70 mm** |
| 3DPW test (14-view pseudo)       | 24 | **387.84 mm** | **88.95 mm** |

For reference, the same checkpoint achieves **~13.84 mm** root-aligned MPJPE on
MPI-INF-3DHP (val S2 Seq1).  The 3DPW pseudo-data error is therefore an order
of magnitude larger.

### Per-sequence root-aligned MPJPE (validation)

| Sequence | Root-aligned MPJPE |
|---|---|
| courtyard_basketball_01 | 47.81 mm |
| courtyard_dancing_00    | 20.33 mm |
| courtyard_drinking_00   | 14.97 mm |
| courtyard_hug_00        | 16.41 mm |
| courtyard_jumpBench_01  | 78.08 mm |
| courtyard_rangeOfMotions_01 | 24.82 mm |
| downtown_walkDownhill_00 | 134.32 mm |
| outdoors_crosscountry_00 | 24.31 mm |
| outdoors_freestyle_01    | 315.78 mm |
| outdoors_golf_00         | 21.73 mm |
| outdoors_parcours_00     | 39.28 mm |
| outdoors_parcours_01     | 242.56 mm |

### Per-sequence root-aligned MPJPE (test, selected outliers)

| Sequence | Root-aligned MPJPE |
|---|---|
| downtown_runForBus_00    | 414.36 mm |
| downtown_walkUphill_00   | 727.55 mm |
| downtown_walkBridge_01   | 261.63 mm |
| downtown_upstairs_00     | 134.35 mm |
| downtown_windowShopping_00 | 95.16 mm |

Most static/downtown sequences lie in the 13–60 mm range, while fast or
heavily occluded sequences (running, uphill walking, freestyle/parcours) blow
up to several hundred millimetres.

## Domain-gap issues observed

1. **Camera / view-count mismatch (resolved by re-conversion).**
   - The checkpoint was trained with 14 calibrated MPI-INF-3DHP cameras.
   - The existing 4-view 3DPW pseudo conversion cannot be used directly because
     the `fusion_mlp` layer has a fixed `d * n_views` input dimension.
   - Re-converting to 14 pseudo views allows the checkpoint to run, but the
     virtual rig is not the same as the studio rig the model learned from.

2. **Global coordinate / scale drift (large raw MPJPE).**
   - Raw MPJPE is dominated by an absolute translation/scale offset between the
     predicted world coordinates and 3DPW ground truth.
   - After pelvis-root alignment, the error drops by an order of magnitude,
     suggesting the model still captures the skeleton shape but not the
     absolute scene scale.

3. **Skeleton topology and joint-count difference.**
   - 3DPW uses 24 SMPL joints; MPI-INF-3DHP uses a 17/28-joint convention.
   - The residual head and weight head are joint-agnostic (they operate
     per-joint on `d`-dim features), so the model can run on 24 joints, but it
     has never seen SMPL joint semantics during training.

4. **In-the-wild vs. studio domain shift.**
   - 3DPW contains outdoor/fast motions, occlusions, and very different
     camera intrinsics.
   - The model is clearly over-fit to the calibrated studio setting:
     relatively slow courtyard sequences have reasonable errors (~15–50 mm),
     while running/acrobat sequences fail dramatically.

## Blockers

- **None for running the evaluation.** The only blocker was the 4-view vs.
  14-view mismatch, which was resolved by re-converting 3DPW with
  `--n_views 14`.

## Files touched / created

- `experiments/eval_residual_3dpw_pseudo.py` — new evaluation script.
- `data/webbridge/3dpw/converted_14views/` — newly generated 14-view pseudo
  conversions for `train`, `validation`, and `test`.
- `outputs/residual_3dpw_pseudo_val.json` — per-sequence validation results.
- `outputs/residual_3dpw_pseudo_test.json` — per-sequence test results.
- `docs/swarm_iter6/residual_3dpw_pseudo_evaluation.md` — this report.

## Recommendations for follow-up

- Train the residual temporal model from scratch (or fine-tune the MPI
  checkpoint) on the 3DPW pseudo-multi-view data to see whether the gap is
  mostly a domain-shift issue or an architectural limitation.
- Investigate per-joint error on the 24 SMPL joints to identify which joints
  transfer poorly.
- Add a learnable / dataset-specific bone-length or scale normalisation to
  reduce the raw MPJPE drift.
