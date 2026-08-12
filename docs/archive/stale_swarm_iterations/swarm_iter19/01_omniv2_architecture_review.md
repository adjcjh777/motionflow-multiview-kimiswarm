# 01_omniv2_architecture_review

## Summary

Review `OmniMultiViewFusionV2` and the running no-graph ablation to assess whether the unified single model can beat the 8.35 mm Bayesian Tri v2 ensemble, and to scope the next iter19 experiments.

## Current state

- **Model:** `motionflow_mv/fusion/omniview_fusion_v2.py` extends `RayAttentionFusionModelBayesianTriV2` with a per-view/per-joint visibility head + fallback guard (lines 131-135, 203-228), a `CrossViewGraphAttention` block (lines 137-145, 280-281), and inherits anisotropic covariance / adaptive Gauss-Newton / residual MLP from the Bayesian Tri v2 parent.
- **Training:** `experiments/train_omniview_fusion_v2_mpiinf3dhp.py` uses `TrainerV2` with view dropout, visibility BCE, uncertainty NLL, temporal velocity, and bone-length losses.
- **Eval:** `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py` runs clean MPJPE/PA-MPJPE, a calibration-robustness matrix, and a variable-view MPJPE@k curve.
- **Ongoing run:** A no-graph ablation (`--graph_num_layers 0`) is training via `scripts/run_omniview_fusion_v2_full_wsl.sh`; target checkpoint `outputs/omniview_fusion_v2_d128_no_graph.pth`, log `outputs/omniview_fusion_v2_d128_no_graph.log`.
- **Earlier run:** `outputs/omniview_fusion_v2_d128.log` (graph layer enabled) only completed the 5-epoch freeze stage (~25 mm val MPJPE) before stopping.

## Key findings

1. **Graph can be cleanly disabled.** `graph_num_layers=0` creates an empty `CrossViewGraphAttention` `ModuleList` (`motionflow_mv/fusion/prototypes/cross_view_graph_attention.py:186`), so the ablation is a single-flag change.
2. **Graph is not view-mask aware.** `VariableViewInferenceWrapper` zeros inactive views’ confidences/pixels, but the graph edge index still connects all `n_views` (`cross_view_graph_attention.py:191-216`). Dropped views remain as graph nodes and may receive messages.
3. **`index_reduce_` is a beta API.** The graph layer uses `index_reduce_(..., reduce="amax", include_self=True)` at `cross_view_graph_attention.py:54`; PyTorch warns the API may change.
4. **Warm-start mismatch is handled.** Loading `bayesian_tri_v2_stabilized_mpiinf3dhp.pth` drops old `joint_attn.0.*` weights and initializes the new visibility/graph buffers (log in `outputs/omniview_fusion_v2_d128_no_graph.log`).
5. **Freeze-stage val MPJPE is high but expected.** The no-graph log shows ~44 mm during the 5-epoch freeze while only the new heads train; the encoder/ST-transformer are still frozen.
6. **Eval script defaults are out of sync.** `scripts/eval_omniview_fusion_v2_wsl.sh` hard-codes `--graph_num_layers 1`, which will mismatch the no-graph checkpoint if used unchanged.
7. **Smoke tests exist.** `tests/test_train_omniview_fusion_v2_smoke.py` and the inline `__main__` smoke in `omniview_fusion_v2.py` provide fast sanity checks.

## Recommendations

1. **Finish and evaluate the no-graph ablation first.** Run `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py` with `--graph_num_layers 0` on full S2/Seq1, including robustness and variable views. This is the highest-value data point for iter19.
2. **Make graph propagation respect active views.** Add a view mask to `CrossViewGraphAttentionLayer.forward` and drop edges whose source node is in an inactive view; otherwise variable-view k=4 targets are unreliable.
3. **Replace the beta `index_reduce_` call.** Use `torch.scatter_reduce` or `torch_scatter` in `_scatter_softmax` to remove a portability/reproducibility risk.
4. **Sync the eval script with the training config.** Pass `graph_num_layers` (and other key hyperparameters) through the eval runner, or load them from the checkpoint manifest.
5. **Run a small paired d=64 smoke: graph vs. no-graph.** Fix seed and train for ~5 epochs on MPI S1/S3 with `graph_num_layers in {0,1,2}` to isolate the graph effect before another full d=128 run.
6. **Use append-mode logging.** The no-graph log appears duplicated at the unfreeze boundary, suggesting a process restart overwrote output. Use `tee -a` or append redirection in the runner.

## Open questions

- Does the no-graph variant reach the 8.0 mm single-model target? If yes, the graph layer may be unnecessary; if no, the graph layer becomes the next lever.
- Is the visibility head learning meaningful occlusion masks, or collapsing to near-1.0 everywhere? A validation histogram would tell.
- How much of the variable-view failure is due to unmasked graph edges versus an under-trained visibility head at low view counts?
- Does the 5-epoch freeze/unfreeze schedule help? A paired ablation with `--warm_start_freeze_epochs 0` would clarify.
