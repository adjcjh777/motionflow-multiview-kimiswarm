# 14 — Model Scaling Plan

## Summary

This subtask covers how to grow (or constrain) the capacity of the unified OmniMultiViewFusionV2 architecture so it beats the current 8.35 mm MPJPE ensemble anchor on MPI-INF-3DHP S2/Seq1 without over-engineering. Earlier naive upsizing produced worse results, so the plan must be evidence-based rather than a blind d→192/d→256 race.

## Current state

- **Running experiment:** a no-graph ablation of `OmniMultiViewFusionV2` trains via `scripts/run_omniview_fusion_v2_full_wsl.sh` with `graph_num_layers=0`, `d=128`, `n_st_layers=3`, `residual_hidden=256` and ~970 k parameters. Its log (`outputs/omniview_fusion_v2_d128_no_graph.log`, lines 6–11) shows the 5-epoch freeze stage is complete and end-to-end training has just started. Freeze-stage validation MPJPE stayed high (~44 mm), expected while only the new heads were trained.
- **Best anchor:** Bayesian Tri v2 ensemble at **8.35 mm** MPJPE / **5.29 mm** PA-MPJPE, composed of two d=128 models (`docs/results_icra_cvpr_2027.md`). The single d=128 stabilized model is at 9.03 mm.
- **Scaling precedent:** a previous 1.06 M-parameter cross-view residual model (`d=128`, `n_st_layers=3`, `residual_hidden=256`) reached only **13.90 / 10.90 mm** (`docs/experiment_log_icra_cvpr_2027.md:42`, `docs/paper_draft_icra_cvpr_2027.md:167`), worse than the 243 k-parameter d=64/h=128 model at 9.32 mm. This warns against naive upsizing.
- **Capacity dimensions available:** `d`, `residual_hidden`, `n_st_layers`, `graph_num_layers`, and training length. The current no-graph run is already at the upper end of the explored width/depth envelope.

## Key findings

1. **The no-graph ablation is the critical gate.** Its outcome determines whether the unified architecture justifies further capacity investment (`docs/swarm_iter18/next_iteration_decision_matrix.md` §2).
2. **Parameter count is not the bottleneck.** The 1.06 M model underperformed the 243 k model, and the current ~970 k no-graph run may already be over-capacity. Adding width or depth without an architectural reason is unlikely to help.
3. **Graph attention is cheap and targeted.** `CrossViewGraphAttention` uses sparse scatter attention over a fixed skeleton graph. If the no-graph run is near the anchor, the first scaling experiment should add `graph_num_layers=1`, not increase `d`.
4. **Training length and augmentation are underexplored.** The current run uses 30 epochs and 10% view dropout. Longer schedules, stronger view dropout, and repeated seeds are lower-risk ways to squeeze accuracy than a larger model.
5. **Scaling infrastructure exists but is not wired for OmniMultiViewFusionV2.** `experiments/prototypes/deeper_st_attention_model/deeper_st_attention_model.py` provides a deeper T×V×J block, and `experiments/run_repeated_seeds.py` exists for reproducibility, but there is no single one-variable capacity sweep script.

## Recommendations

1. **Wait for the no-graph ablation to finish, then decide.** Do not launch `d=192/d=256` runs until `outputs/omniview_fusion_v2_d128_no_graph.pth` is evaluated. Use the decision matrix in `docs/swarm_iter18/next_iteration_decision_matrix.md` §2.
2. **If no-graph clean MPJPE < 8.35 mm:** keep d=128 and add `graph_num_layers=1` in a separate run. Only after graph attention is evaluated should a capacity sweep (d{96,160}, h∈{128,256}) be considered.
3. **If no-graph clean MPJPE is 8.35–9.0 mm:** run a focused ablation grid one variable at a time: `n_st_layers∈{2,3}`, `residual_hidden∈{128,256}`, `graph_num_layers∈{0,1}`. Log each in `outputs/omniview_v2_ablation_<name>/` with a `manifest.json`.
4. **If no-graph clean MPJPE > 9.0 mm:** pause OmniMultiViewFusionV2 scaling. Fall back to the Bayesian Tri v2 stack and apply proven improvements incrementally, per `docs/swarm_iter18/next_iteration_decision_matrix.md` §3.1.
5. **Add a runtime/parameter budget guard.** Reuse `experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py` to report parameters, peak memory, and clips/s. Any scaled-up model must stay within 1.3× the wall-time of the d=128 no-graph baseline.
6. **Make repeated seeds a gating requirement.** Any new anchor must pass ≥3 seeds via `experiments/run_repeated_seeds.py` and report mean/std, per `docs/swarm_iter18/next_iteration_decision_matrix.md` §5.

## Open questions

- Does the no-graph ablation reach < 8.35 mm, and does adding graph attention help or hurt?
- Is `d=128/n_st_layers=3/residual_hidden=256` already over-capacity, or did the prior 1.06 M model fail for a different reason?
- How much of the 8.35 mm ensemble gain comes from ensembling rather than single-model capacity?
- What is the wall-time and memory of the ~970 k no-graph model versus the Bayesian Tri v2 baseline?
- Should we scale by longer training (40–50 epochs) and stronger view dropout before touching width/depth?
