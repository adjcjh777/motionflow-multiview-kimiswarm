# 09 WebBridge MPI-INF-3DHP Splits

## Summary

This subtask audits how the MotionFlow-MultiView repository defines, converts,
and loads the **MPI-INF-3DHP** train/val/test splits through the WebBridge
pipeline. Reproducible, well-documented splits are a publication blocker: the
ICRA/CVPR 2027 story relies on an clean train→val→test protocol, repeated-seed
runs, and an official test-set (TS1–TS6) submission path. We reviewed the
canonical `.npz` inventory, the manifest/audit scripts, the test-set converter,
and the downstream loaders.

## Current state

**What exists**

- **Canonical training/validation data**: 43 `.npz` files under
  `data/webbridge/mpi_inf_3dhp/` covering MPI-INF-3DHP subjects S1–S8. Most
  sequences are available in 14-view and 4-view variants, with and without the
  `_m` (meters) tag (`motionflow_mv/data/webbridge_loader.py:347`).
- **Test-set conversion**: TS1–TS6 have been converted to
  `data/webbridge/mpi_inf_3dhp/test_set/TS{i}_v14_multiview.npz`
  (`experiments/prototypes/swarm_iter18/convert_mpiinf3dhp_test_set.py`).
- **Audit tooling**:
  `experiments/prototypes/audit_webbridge_mpiinf3dhp_data.py` scans the main
  directory and emits a JSON/CSV manifest of coverage and missing sequences.
  The tests in `tests/test_webbridge_mpiinf3dhp_manifest.py` pass (4/4).
- **Hard-coded split logic**: several training/evaluation scripts encode the
  split by path, e.g. `experiments/run_publishable_benchmark_plan.py:49` uses
  S1-Seq1/Seq2 for training, S2-Seq1 for validation, and S3-Seq1 as a hold-out
  “test” proxy.

**What does not work / is incomplete**

- **Missing training sequence**: S2 Seq2 is absent. The raw directory only has
  `raw/S2/Seq1`; the batch converter therefore cannot produce
  `s_02_seq_02_v14_multiview_m.npz`.
- **Test-set not recognized by the auditor**: the audit script expects test
  files named `TS{i}_v14_multiview_m.npz` in the *same* directory as the
  training files. The actual converted files live in `test_set/` and lack the
  `_m` tag, so they are reported as missing.
- **Skeleton mismatch with the mixed loader**:
  `motionflow_mv/data/webbridge_mixed_dataset.py:71` maps the 28-joint
  MPI-INF-3DHP skeleton down to 17 joints. The official test set already uses
  a 17-joint skeleton, so the mixed loader cannot currently consume it.
- **No single source-of-truth split config**: splits are repeated as string
  paths across many scripts; there is no YAML or manifest that formally lists
  the official train/val/test split.

## Key findings

1. **Coverage gap in training data** (`outputs/webbridge_mpi_inf_3dhp_manifest.json`)
   - 43 files scanned, all valid, but S2 Seq02 is flagged as missing.
   - The missing raw data is at `data/webbridge/mpi_inf_3dhp/raw/S2/Seq2/`
     (directory does not exist).

2. **Test-set files are valid but separate**
   - `data/webbridge/mpi_inf_3dhp/test_set/TS1_v14_multiview.npz` has shape
     `(6151, 14, 17, 2)` for `points_2d`; the other TS files are similarly
     shaped.
   - `valid_frame` from `annot_data.mat` is mapped to the confidence mask, as
     documented in `docs/mpiinf3dhp_test_set_conversion.md`.

3. **Audit script filename assumptions**
   - `experiments/prototypes/audit_webbridge_mpiinf3dhp_data.py:41` only matches
     filenames of the form `s_<subject>_seq_<seq>_v<views>_multiview_<tags>.npz`.
     It does not match `TS{i}_v14_multiview.npz`, nor does it recurse into
     `test_set/`.

4. **Loader split by file path is brittle**
   - `experiments/run_publishable_benchmark_plan.py:49` hard-codes the train/val
     split. Any change to the MPI directory or missing sequence requires editing
     every script that references these paths.

## Recommendations

1. **Restore S2 Seq02**
   - Run `python experiments/batch_convert_mpiinf3dhp_v1.py --subjects 2
     --sequences 2 --download --yes-download` to fetch the missing raw files,
     then convert both 14-view and 4-view variants.

2. **Unify test-set handling in the auditor**
   - Update `experiments/prototypes/audit_webbridge_mpiinf3dhp_data.py` to also
     scan `data/webbridge/mpi_inf_3dhp/test_set/` and recognize test filenames.
   - Optionally rename test outputs to `TS{i}_v14_multiview_m.npz` and place
     them under the main directory so the existing filename regex works, or
     update the regex to support the current names.

3. **Create a split manifest/YAML**
   - Add `configs/splits/mpiinf3dhp_train_val_test.yaml` (or extend
     `configs/benchmark_webbridge_mpi_smoke.yaml`) that explicitly lists the
     official train, val, and test files. This should become the single source
     of truth for all MPI-INF-3DHP experiments.

4. **Make the mixed loader test-set aware**
   - In `motionflow_mv/data/webbridge_mixed_dataset.py`, detect 17-joint MPI
     files and bypass the 28→17 skeleton mapping for them. Alternatively, keep
     test-set inference separate (as it is now in
     `experiments/infer_mpiinf3dhp_test_set_omniview_v2.py`) and document this
     design decision.

5. **Add regression tests**
   - Extend `tests/test_webbridge_mpiinf3dhp_manifest.py` to assert that the
     generated manifest matches the expected train/val/test split YAML and
     that no expected sequence is missing after step 1.

## Open questions

- Is the 17-joint test-set skeleton identical to the canonical 17-joint layout
  used by H36M/AIST++ in `motionflow_mv/data/webbridge_mixed_dataset.py`? A
  joint-name check is needed before merging test-set training with other
  datasets.
- How much does the missing S2 Seq02 affect model performance? Should it be
  treated as an expected gap or a blocker for the final benchmark?
- Do we want the WebBridge mixed loader to consume the official test set for
  **self-supervised pre-training**, or keep it strictly inference-only to
  preserve the official evaluation protocol?
