# 06 — Temporal / Spatio-Temporal Transformer

## Summary

This subtask covers the **temporal / spatio-temporal (ST) transformer** that fuses multi-view pose evidence across time and views inside `OmniMultiViewFusionV2`. The current design attends jointly over `(time × view)` tokens per joint, bridging per-frame ray-aware features and final uncertainty-weighted triangulation. The goal is to verify that this block is sound, identify the most promising ST variant, and give the next iteration a concrete ablation plan.

## Current state

* **Three ST variants exist in the tree:**
  * `(T × V)` per-joint transformer in `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py:91-98` — used by `OmniMultiViewFusionV2`.
  * Factorised `(T × V × J)` transformer in `motionflow_mv/models/spatiotemporal_principal_point_model.py:135-262`.
  * Unified `(T × V × J)` grid transformer in `motionflow_mv/fusion/ray_attention_spatiotemporal_model.py:99-107`.
* **OmniMultiViewFusionV2 wiring:** graph-joint attention feeds into the `(T × V)` transformer at `motionflow_mv/fusion/omniview_fusion_v2.py:280-292`.
* **Temporal consistency tooling:** `motionflow_mv/losses/temporal_consistency_v2.py` provides velocity/acceleration Huber loss with visibility masking; smoke tests pass.
* **Ongoing run:** a `graph_num_layers=0` ablation is training via `scripts/run_omniview_fusion_v2_full_wsl.sh`. Its log (`outputs/omniview_fusion_v2_d128_no_graph.log`) shows five frozen-encoder warmup epochs ending at ~44.4 mm val MPJPE; end-to-end training just started.
* **Prior run:** the full `d=128` run with graph attention (`outputs/omniview_fusion_v2_d128.log`) reported val MPJPE ~25 mm during the frozen stage.
* **Tests:** `pytest tests/test_spatiotemporal_pp_model.py tests/test_ray_attention_spatiotemporal.py tests/test_temporal_consistency_v2.py tests/test_train_omniview_fusion_v2_smoke.py` passes (20/20).

## Key findings

1. **The active ST block is a standard `nn.TransformerEncoderLayer` stack over flattened `(T·V, d)` tokens per joint** (`omniview_fusion_v2.py:289-292`). For default `T=13` and `V=14`, each joint attends over 182 tokens. This is the highest-cost part of the model and where deeper or factorised alternatives matter most.
2. **Warm-start discards old dense joint-attention weights and initialises new heads**: logs list `joint_attn.*` as unexpected and `visibility_head.*`/`graph_joint_attention.*` as missing. The frozen-encoder phase therefore shows high MPJPE while the new heads learn in isolation; do not judge final quality from these numbers.
3. **The no-graph ablation cleanly isolates the ST transformer + visibility/uncertainty heads from graph-joint attention.** If it reaches the ~9 mm neighborhood of the Bayesian Tri v2 anchor, the ST block itself is valuable. If not, the bottleneck likely lies in feature preparation before the transformer.
4. **Alternative ST architectures are prototype-only.** `SpatiotemporalPrincipalPointModel` and `DeeperStAttentionPrincipalPointModel` (`experiments/prototypes/deeper_st_attention_model/deeper_st_attention_model.py`) have no GPU training results yet.

## Recommendations

1. **Wait for the no-graph ablation to finish, then evaluate it** with `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py --run_robustness --run_variable_views`. This measures the ST block's real contribution.
2. **If clean MPJPE is not within ~10 % of the 9.03 mm single-model anchor,** replace the `(T × V)` block in `omniview_fusion_v2.py:283-292` with the factorised `T × V × J` layers from `spatiotemporal_principal_point_model.py`. Smaller per-axis token counts may train more stably.
3. **Add `TemporalConsistencyLossV2` with visibility masking** to the trainer (`train_omniview_fusion_v2_mpiinf3dhp.py:423-429`). The current velocity-only term ignores the predicted visibility mask; masking will make the temporal loss robust to occluded frames.
4. **Run a short d=48 GPU smoke comparing the three ST variants** (T×V, factorised T×V×J, unified grid) on MPI-INF-3DHP S1/S2 with identical warm-start, freeze schedule, clip_len 13, and seed.

## Open questions

* Does the current `(T × V)` ST transformer alone (no graph attention) improve over the Bayesian Tri v2 single model, or does most of the gain come from visibility/uncertainty heads?
* Is the 5-epoch frozen-encoder warmup optimal? The high freeze-stage MPJPE suggests the new heads may need more warmup or a higher initial learning rate.
* Which ST variant gives the best accuracy/memory trade-off for the final OmniMultiViewFusionV2? The factorised `T × V × J` design has the best scaling but is untrained.
