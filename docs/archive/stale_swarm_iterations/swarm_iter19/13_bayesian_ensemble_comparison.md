# 13 Bayesian Ensemble Comparison

## Summary

This subtask compares the current best **Bayesian Tri v2 ensemble** (8.35 mm MPJPE on MPI-INF-3DHP S2/Seq1) against the in-flight **no-graph ablation of OmniMultiViewFusionV2** and the newly introduced **uncertainty-aware ensemble inference v2** harness. The goal is to clarify whether the new architecture/aggregation can beat the 8.35 mm anchor and to identify the fastest path to a new best result.

## Current State

- **Anchor ensemble** — `outputs/bayesian_tri_v2_ensemble_2_eval.json` reports **8.35 mm** MPJPE / **5.29 mm** PA-MPJPE / **0.9444** PCK-AUC (`docs/swarm_iter18/P01_state_audit.md:13`, `docs/experiment_log_icra_cvpr_2027.md:22-24`). It combines `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth` (single 9.03 mm) and `outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth`. The ensemble is produced by `scripts/eval_ensemble_wsl.sh`, which calls `experiments/prototypes/eval_ensemble_checkpoints.py` and performs a simple uniform (or weighted) average of member predictions.
- **Uncertainty-aware ensemble v2** — `experiments/prototypes/swarm_iter18/ensemble_inference_v2.py:57-248` implements `BayesianTriV2Ensemble`, supporting `uniform`, `inverse_variance`, `robust_median`, and `trimmed_mean` aggregation. It is documented in `docs/swarm_iter18/P09_ensemble_inference_v2.md` and has passed CPU smoke tests, but it has not yet been run on the full S2/Seq1 validation set.
- **No-graph ablation** — `scripts/run_omniview_fusion_v2_full_wsl.sh:28` trains `OmniMultiViewFusionV2` with `--graph_num_layers 0`, warm-started from `bayesian_tri_v2_stabilized_mpiinf3dhp.pth` and freezing the encoder/ST-transformer for 5 epochs. The current log `outputs/omniview_fusion_v2_d128_no_graph.log:7-11` shows the model is still in the freeze stage with validation MPJPE around **44 mm** (epoch 1: 46.93 mm → epoch 5: 44.37 mm), far from convergence.

## Key Findings

1. **The 8.35 mm anchor is a two-model uniform ensemble**, not a single model. The single stabilized member is already at 9.03 mm (`docs/results_icra_cvpr_2027.md:12`), so most of the gain over single-model performance comes from averaging the stabilized and augmented checkpoints.
2. **`ensemble_inference_v2.py` is ready to test but unvalidated on the anchor.** It reuses `experiments/eval_full_metrics.py` for data loading and adds per-joint inverse-variance weighting using the Cholesky covariance `L` returned by `RayAttentionFusionModelBayesianTriV2` (`ensemble_inference_v2.py:141-174`). Whether this improves on a plain uniform average is still empirical.
3. **The no-graph ablation is early and not yet competitive.** After 5 frozen epochs its validation MPJPE is ~44 mm. It needs to complete the planned 30 epochs and unfreeze before a fair comparison is possible.
4. **The existing ensemble script has a known footgun:** `eval_ensemble_checkpoints.py` defaults to `d=64`, but the anchor checkpoints are `d=128`. If `scripts/eval_ensemble_wsl.sh` is invoked without the correct `--d 128 --residual_hidden 256 --n_st_layers 3`, it fails with a size-mismatch error (`docs/swarm_iter18/P01_state_audit.md:193`).

## Recommendations

1. **Run the new ensemble v2 on the anchor checkpoints.** Execute `ensemble_inference_v2.py` with `--strategy inverse_variance` (and also `uniform` as a baseline) against the existing two-member ensemble. This is the fastest way to validate whether uncertainty-aware aggregation can push the anchor below 8.35 mm.
2. **Wait for the no-graph ablation to finish, then evaluate it fairly.** Use `scripts/eval_omniview_fusion_v2_wsl.sh` with `--graph_num_layers 0` and the same `val_stride=50` protocol. Compare clean MPJPE/PA-MPJPE directly against `bayesian_tri_v2_stabilized_mpiinf3dhp.pth` (9.03 mm single) and the 8.35 mm ensemble.
3. **Fix the ensemble evaluation script defaults.** Update `scripts/eval_ensemble_wsl.sh` to explicitly pass `--d 128 --residual_hidden 256 --n_st_layers 3` so the 8.35 mm result is reproducible without manual flags.
4. **Add a strategy-sweep helper.** A small wrapper that runs `ensemble_inference_v2.py` with `uniform`, `inverse_variance`, `robust_median`, and `trimmed_mean` on the same checkpoints would make it easy to decide which aggregation to adopt as the new standard.

## Open Questions

- Does `inverse_variance` aggregation actually improve over the current uniform average, or is the anchor already near the ensemble ceiling?
- Will the no-graph OmniMultiViewFusionV2 ablation reach or beat the 8.35 mm ensemble once fully trained, or does the graph-joint attention provide a non-trivial gain?
- How does the no-graph model's robustness (noise, view dropout, calibration perturbation) compare to the Bayesian Tri v2 ensemble? A clean-MPJPE win that sacrifices robustness is not clearly better for the paper.
- If `ensemble_inference_v2.py` helps, can additional members (e.g., `bayesian_tri_v2_visibility_mpiinf3dhp.pth`, `bayesian_tri_v2_attention_entropy_mpiinf3dhp.pth`) be added to reach the stated target of **≤ 7.8 mm**?
