# 12 — Robustness Matrix Evaluation

## Summary

This subtask covers the standardized robustness-evaluation harness used to gate
anchor-model decisions. A robustness matrix evaluates a trained checkpoint under
clean conditions and a fixed grid of input/calibration degradations. The
results determine whether a new model is both accurate *and* deployable enough to
replace the current 8.35 mm Bayesian Tri v2 ensemble anchor.

The repo currently has **three partially overlapping matrix scripts**:

- `experiments/run_robustness_matrix.py` — basic 2-D / view-dropout / PP matrix.
- `experiments/prototypes/run_extended_robustness_matrix.py` — extended noise /
  joint occlusion / view-dropout matrix with pairwise/three-way combos and a
  built-in CPU smoke test.
- `experiments/eval_perturb_model_mpiinf3dhp.py` and `experiments/prototypes/swarm_iter18/eval_robustness_matrix_v2.py`
  — calibration-robustness (rot/trans/focal/PP) plus occlusion for the ray-attention
  / OmniMultiViewFusion v2 family.

## Current state

- The **extended matrix** is the most complete and has been run on the current
  best single-model checkpoint (`bayesian_tri_v2_stabilized_mpiinf3dhp.pth`).
  Results are in
  `outputs/extended_robustness_matrix_bayesian_tri_v2_stabilized/robustness_matrix.json`
  and `.md`.

- The **basic matrix** has been run on a previous checkpoint in
  `outputs/robustness_matrix_epipolar_bias_v2_pp/`.

- The **OmniMultiViewFusion v2 no-graph ablation**
  (`outputs/omniview_fusion_v2_d128_no_graph.pth`) is still training. Its log
  shows the 5-epoch freeze phase finishing at ~44.4 mm val MPJPE
  (`outputs/omniview_fusion_v2_d128_no_graph.log`, lines 6–11), so no robustness
  matrix has been produced for it yet.

- The gating rules in `docs/iter_next_action_plan.md` (lines 109–113) require
  `rot_0.5° < 12 mm`, `focal_1% < 14 mm`, and `pp_10px < 50 mm`.

## Key findings

1. **Strong Gaussian-noise robustness on the current best model.**
   The extended matrix on Bayesian Tri v2 stabilized shows almost no degradation
   from 2 px keypoint noise: MPJPE rises only from **9.03 mm (clean)** to
   **9.31 mm** (`outputs/extended_robustness_matrix_bayesian_tri_v2_stabilized/robustness_matrix.md`,
   lines 3–6). This is a publishable data point.

2. **View dropout is the dominant failure mode.**
   50 % view dropout degrades MPJPE to **23.89 mm** and drops PCK@50 to 0.949,
   while 30 % view dropout raises MPJPE to **18.15 mm** (extended matrix, lines
   9–12). Occlusion is secondary: 30 % joint occlusion reaches **16.99 mm**.

3. **Calibration robustness is still the largest gap versus publication gates.**
   The best published calibration matrix (`docs/results_icra_cvpr_2027.md`,
   lines 23–34) reports `rot_0.5° = 16.89 mm` and `focal_1% = 19.13 mm`, both
   failing the gating thresholds. Principal-point perturbation remains
   catastrophic in earlier experiments.

4. **Older `epipolar_bias_v2_pp` matrix shows principal-point sensitivity.**
   `outputs/robustness_matrix_epipolar_bias_v2_pp/robustness_matrix.md` gives
   clean 27.73 mm, 10 px PP 30.23 mm, and 2 px noise 27.79 mm, confirming PP
   drift is harder than keypoint noise.

5. **Tooling is fragmented.**
   `experiments/prototypes/run_extended_robustness_matrix.py:58-90` defines the
   noise/occlusion/dropout grid. `experiments/run_robustness_matrix.py:62-72`
   defines a different, smaller grid. There is no single script that combines
   calibration perturbations with the extended input-degradation matrix.

## Recommendations

1. **Run the extended matrix on the no-graph ablation as soon as its checkpoint
   is ready.** Use the existing runner:
   ```bash
   scripts/run_extended_robustness_matrix_wsl.sh \
       omniview_fusion_v2_d128_no_graph \
       outputs/omniview_fusion_v2_d128_no_graph.pth \
       data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz
   ```
   This directly answers whether dropping graph attention hurts noise /
   occlusion / view-dropout robustness.

2. **Converge the two matrix scripts into one call path.** Add an
   `--extended_robustness` flag to `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py`
   that dispatches `experiments/prototypes/run_extended_robustness_matrix.py`,
   so every OmniMultiViewFusion v2 ablation emits both calibration and 2-D
   degradation matrices in a single command.

3. **Standardize the output schema.**
   `run_extended_robustness_matrix.py` stores conditions under `"conditions"`
   while `eval_perturb_model_mpiinf3dhp.py` uses `"robustness"`. Pick one key
   and update downstream summary scripts accordingly.

4. **Update the gating table in `docs/iter_next_action_plan.md:109-113`** once
   no-graph numbers arrive. If the no-graph model fails the view-dropout gate
   but passes calibration gates, that is the decisive signal for whether graph
   attention is worth its cost.

## Open questions

- Does the no-graph ablation lose occlusion robustness relative to the graph-
  enabled baseline? The extended matrix will quantify this.
- Is the Bayesian Tri v2 model’s excellent noise robustness preserved in the
  single-model OmniMultiViewFusion v2 architecture?
- Will the PP-correction layer in OmniMultiViewFusion v2 finally make
  `pp_10px` robustness non-catastrophic?
