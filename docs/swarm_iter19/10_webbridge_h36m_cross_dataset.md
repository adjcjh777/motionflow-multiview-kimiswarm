# 10_webbridge_h36m_cross_dataset

## Summary

This subtask covers the **Human3.6M (H36M) leg of the WebBridge cross-dataset benchmark**. The goal is to validate that the project’s canonical `.npz` pipeline, 17-joint mixed loader, and evaluation harness can produce reliable H36M numbers, and to identify what is blocking a clean MPI→H36M / H36M→MPI cross-dataset story for the ICRA/CVPR 2027 paper. H36M is the largest, most widely cited reference dataset available in WebBridge, so its inclusion is essential for any credible cross-dataset claim.

## Current state

**Data assets are in place and audited.**

- Canonical H36M files exist in both millimeter (`data/webbridge/h36m/`) and meter (`data/webbridge/h36m_meters/`) variants, covering S1 train and S9/S11 test actions (`docs/swarm_iter18/P07_webbridge_manifest.md`, §3).
- The WebBridge quality audit reports **45 H36M files, all canonical**, with 4 views and 17 joints (`docs/swarm_iter18/P07_webbridge_manifest.md`).
- The 17-joint unification path is implemented in `motionflow_mv/data/webbridge_mixed_dataset.py` (`SKELETON_MAPS`, `CANONICAL_17_JOINTS`, `WebBridgeCanonical17Dataset`), which pads small-view rigs to 14 views and re-indexes MPI 28→17 joints.
- A reusable benchmark harness exists: `experiments/run_webbridge_benchmark.py` reads a YAML manifest, calls `experiments/eval_full_metrics.py` per dataset, and writes CSV/JSON summaries.

**Existing H36M numbers (single-dataset baselines).**

- A dedicated H36M-trained principal-point residual model (`outputs/ray_attention_temporal_crossview_residual_principal_point_h36m_full.pth`) reaches **5.24 mm MPJPE / 4.84 mm PA-MPJPE** on a held-out H36M test (`outputs/crossview_pp_h36m_full_eval.json`).
- A smaller H36M residual model reached 5.71 mm val MPJPE (`outputs/train_h36m_h64.log`).
- A tiny MPI+H36M+AIST mixed-residual experiment reported **7.08 mm MPJPE on H36M** (`outputs/mixed_residual_h36m_train_eval.json`), but this was a small-scale run, not a full benchmark.

**The cross-dataset benchmark has not yet been executed cleanly for H36M.**

- `configs/benchmark_webbridge_h36m_test_smoke.yaml` lists every S9/S11 action and points to the MPI-trained 14-view checkpoint `outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth`.
- **The config omits `source_n_views: 14`** (line 2–10). `eval_full_metrics.py` then builds a 4-view model for H36M and loads the 14-view checkpoint with `strict=False`, which is almost certainly incorrect and likely to fail or produce garbage.
- No H36M benchmark CSV/JSON has been written to `outputs/` by the WebBridge harness.
- `outputs/webbridge_benchmark_crossview_residual_full.log` is empty (0 bytes), confirming a prior full run never completed.

## Key findings

1. **View-count mismatch is the immediate blocker.** The H36M smoke manifest does not set `source_n_views`, so the harness will not use `VariableViewInferenceWrapper` (`experiments/eval_full_metrics.py:234–245`). The MPI-trained 14-view model cannot be evaluated on 4-view H36M without that wrapper and the `strict=False` load will drop most weights.

2. **Mixed-loader smoke runs but shows a scale/dataset red flag.** `outputs/webbridge_mixed_smoke.log` shows a tiny training loss (0.0016) but a **validation loss of ~408**. This is consistent with mixing meter-scaled H36M with millimeter-scaled MPI/AIST in the same batch; the current mixed loader does not enforce a common world unit. `motionflow_mv/data/webbridge_mixed_dataset.py` only re-indexes joints and pads views, it does **not** rescale 3D coordinates or cameras across datasets.

3. **H36M conversion already supports the correct meter variant.** `experiments/batch_convert_h36m_webbridge.py` can emit `_m.npz` files (line 100–111), and the smoke config uses `data/webbridge/h36m_meters/`, which is correct. The problem is downstream: other WebBridge assets (MPI/AIST) and some H36M experiments are still in millimeters, so a unified meter-scaled manifest is incomplete.

4. **The 17-joint skeleton map for MPI is hard-coded and lightly validated.** `motionflow_mv/data/webbridge_mixed_dataset.py:71–74` maps MPI 28 joints to H36M 17 joints. There is no runtime reprojection-quality check after re-indexing, so mapping errors would silently bias cross-dataset numbers.

5. **Cross-dataset transfer has not been measured.** There is no JSON/CSV yet that reports MPI-only model performance on the full H36M S9/S11 test set, nor H36M-trained model performance on MPI. `outputs/cross_dataset_generalization_test.json` shows catastrophic transfer errors (~748 mm target MPJPE), but that file uses the old `ray_attention_v3_h36m.pth` checkpoint and `data/h36m_hf/`, not the canonical WebBridge pipeline.

## Recommendations

1. **Fix the H36M benchmark config and run it.** Add `source_n_views: 14` to `configs/benchmark_webbridge_h36m_test_smoke.yaml` (and to any full H36M manifest), then run:
   ```bash
   python experiments/run_webbridge_benchmark.py \
       --manifest configs/benchmark_webbridge_h36m_test_smoke.yaml \
       --out outputs/webbridge_benchmark_h36m_test_smoke
   ```
   This is the fastest way to get the first genuine cross-dataset number for H36M.

2. **Audit world-unit consistency before any mixed-dataset training.** Ensure every canonical `.npz` consumed together is in **meters**. Either regenerate all MPI and AIST `.npz` files as `_m.npz` or add an automatic `scale_factor` field to the loader. The current mixed loader silently assumes compatible units.

3. **Add a reprojection-quality gate for the MPI 28→17 joint map.** In `motionflow_mv/data/webbridge_mixed_dataset.py`, after re-indexing, verify that triangulating the mapped 2D points against the ground-truth 3D matches `joints_3d` within a small tolerance for a few sampled frames. This protects the cross-dataset numbers from a silent mapping bug.

4. **Produce the two canonical transfer tables.**
   - **Zero-shot:** MPI-only `crossview_residual_pp` / `bayesian_tri_v2_pp` on all H36M S9/S11 actions.
   - **H36M-only to MPI:** H36M-trained principal-point model on MPI-INF-3DHP S2/Seq1.
   Store outputs under `outputs/webbridge_benchmark_*.csv` and summarize them with `experiments/summarize_webbridge_benchmark.py`.

5. **Extend `mixed_dataset.py` to support shelf/campus in the registry.** `motionflow_mv/data/mixed_dataset.py:27–31` only registers `mpi`, `aist`, and `h36m`. Shelf (5 views) and Campus (3 views) are missing, which blocks the full five-dataset mixed training planned in `docs/swarm_iter18/P18_cross_dataset_plan.md`.

## Open questions

- Does adding `source_n_views: 14` to the H36M manifest produce stable MPJPE numbers, or does the variable-view wrapper introduce other shape/attention issues on 4-view inputs?
- Are the MPI `_m.npz` files actually in meters? If so, can we safely switch the existing MPI smoke manifests to the meter variants and reproduce the 8.35 mm benchmark?
- What is the cross-dataset gap for the current best single-dataset model (`bayesian_tri_v2_pp`) on the full H36M S9/S11 set? The dedicated H36M baseline is 5.24 mm, but the MPI→H36M number is still unknown.
- How much of the 408 validation loss in the mixed-loader smoke is unit mismatch versus a real model-generalization issue? A controlled ablation with all datasets converted to meters is needed.
