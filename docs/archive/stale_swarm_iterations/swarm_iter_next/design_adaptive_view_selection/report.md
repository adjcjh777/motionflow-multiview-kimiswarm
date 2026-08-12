# Adaptive View Selection for MotionFlow-MultiView

## Summary

Adaptive view selection makes the fusion network explicitly choose which cameras contribute to each joint at each frame, rather than softly weighting all views. The current best model (`RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`, `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py:126`) already predicts per-view log-variance and soft DLT weights, but it still consumes every view. A learned selection module can drop redundant or corrupted views, reduce inference cost, and improve robustness when cameras are occluded, poorly calibrated, or geometrically weak for a joint.

## Key Design Decisions

1. **Per-view, per-joint selector.** After the spatio-temporal encoder, pool each `(view, joint)` token to a score and sample a binary selection mask. This is finer-grained than frame-level selection and matches the existing per-view weighting semantics.
2. **Differentiable Gumbel-softmax with straight-through.** Hard selection is non-differentiable; use Gumbel-softmax sampling during training and straight-through hard masks during inference (`torch.argmax` for top-`k`).
3. **Budgeted selection.** Impose a soft or hard budget `k ≤ V` views per joint to guarantee triangulation stability and reduce compute. The budget can be fixed or predicted from scene difficulty.
4. **Geometry-aware score.** Augment the learned token with ray-angle and baseline features so the selector prefers views with wide baselines and non-degenerate rays.
5. **Plug into existing triangulation.** Selected views become a multiplicative mask on the existing uncertainty weights; DLT/GN/residual stages remain unchanged, preserving the 11.17 mm baseline path.

## Equations

Per-view selection score:
\[s_{v}^{(j)} = \mathrm{MLP}\bigl[f_{v}^{(j)} \|\, g_{v}^{(j)}\bigr]\]
where \(f_{v}^{(j)}\) is the encoder token and \(g_{v}^{(j)}\) are ray-angle/baseline geometry features.

Gumbel-softmax mask (training, temperature \(\tau\)):
\[m_{v}^{(j)} = \frac{\exp((s_{v}^{(j)} + \epsilon_{v})/\tau)}{\sum_{u}\exp((s_{u}^{(j)} + \epsilon_{u})/\tau)}, \quad \epsilon_{v} \sim \mathrm{Gumbel}(0,1)\]

Inference: hard top-`k` mask
\[\hat{m}_{v}^{(j)} = \mathbb{1}\bigl[s_{v}^{(j)} \in \mathrm{top\_k}(s^{(j)})\bigr]\]

Masked triangulation weights:
\[w_{v}^{(j)} = \mathrm{conf}_{v}^{(j)} \, \exp(-\log\sigma_{v}^{(j)}) \, \hat{m}_{v}^{(j)}\]

Budget loss (optional) encourages selecting close to \(k\) views:
\[\mathcal{L}_{\mathrm{budget}} = \bigl(\sum_{v} \hat{m}_{v} - k\bigr)^2\]

## Existing Building Blocks

- `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py` — latest combined model; the natural insertion point is after the spatio-temporal transformer (`feat = ...reshape(B * J, T*V, self.d)`) and before uncertainty/weight heads.
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py` — current best baseline (11.17 mm MPJPE) and residual head.
- `docs/swarm_iter11_variable_view_count_report.md` — variable-view groundwork (`max_n_views`, sliced embeddings, view masking).
- `docs/swarm_iter7/occlusion_aware_fusion_mechanisms.md` — occlusion-aware masking shares the same mask-multiply pattern.
- `docs/swarm_iter7/uncertainty_aware_per_view_weighting.md` — existing uncertainty-weighted triangulation.
- `experiments/eval_residual_robustness_mpiinf3dhp_v1.py` — robustness harness to evaluate selection under occlusion/outliers.

## Proposed Next Steps

1. Add `AdaptiveViewSelector` module in `motionflow_mv/fusion/adaptive_view_selector.py`.
2. Create `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_adaptive_v1_model.py` subclassing the combined model, inserting the selector before the uncertainty head.
3. Add training script `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_adaptive_v1_mpiinf3dhp.py`, mirroring the existing combined-model trainer with budget loss and Gumbel-softmax straight-through.
4. Evaluate with `experiments/eval_residual_robustness_mpiinf3dhp_v1.py` augmented to sweep `k ∈ {2, 3, 4}` and occlusion rates.

## Success Metrics

- MPI-INF-3DHP S1→S2/Seq1 MPJPE ≤ 11.17 mm at `k=4` (no regression).
- At `k=2` or 30% occlusion, MPJPE improves ≥5% over baseline.
- Selection mask correlates with true corrupted/occluded views (mask accuracy ≥ 75%).
