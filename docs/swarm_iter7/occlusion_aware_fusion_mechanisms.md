# Occlusion-aware fusion mechanisms

**Summary**

I investigated occlusion-aware fusion for the current best model, `RayAttentionFusionModelTemporalResidual`. The model already learns per-view, per-joint weights and is robust to random joint occlusion, but it does not explicitly predict visibility/occlusion. I propose adding an explicit occlusion-aware branch to the temporal-residual architecture, training it with auxiliary occlusion supervision, and evaluating it with the existing MPI-INF-3DHP robustness harness.

Because this subagent is restricted to read-only exploration, the report below is the content that should be placed at:

`docs/swarm_iter7/occlusion_aware_fusion_mechanisms.md`

---

# Occlusion-aware fusion mechanisms

## 1. Current state

The best architecture is `RayAttentionFusionModelTemporalResidual` in `motionflow_mv/fusion/ray_attention_temporal_residual_model.py:38`. It predicts per-view, per-joint soft weights, multiplies them by the detector confidence, and feeds a differentiable weighted-DLT triangulator (`ray_attention_temporal_residual_model.py:161-163`). A lightweight residual MLP then corrects the raw triangulated pose.

Occlusion is currently handled only implicitly:
- Training-time dropout zeroes out random `(view, joint)` confidences, simulating occlusion (`experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py:111-118`).
- The robustness table in `docs/paper_draft_icra_cvpr_2027.md:96-104` shows 50% random joint occlusion barely changes MPJPE (11.17 mm → 11.18 mm).
- Related but separate work exists:
  - Uncertainty-weighted DLT head (`docs/swarm_iter5/uncertainty_weighted_triangulation.md`).
  - Bone-length/skeleton consistency losses (`docs/swarm_iter5/bone_length_skeleton_consistency_loss.md`).
- Failure analysis (`docs/swarm_iter5/failure_modes.md`) shows **26.3% of high-error frames** are attributed to occlusion.

## 2. Gap / opportunity

The residual model has no explicit occlusion/visibility reasoning:
- It cannot distinguish “low detector confidence” from “joint is actually occluded in this view.”
- It has no hard mask or fallback when a joint is occluded in too many views.
- It does not use auxiliary occlusion labels or a skeleton prior to fill in for missing views.

Adding an explicit occlusion-aware branch would make the fusion more interpretable and potentially improve the already strong robustness numbers.

## 3. Concrete next step

Add an occlusion-aware gating head to the temporal-residual model:

1. Create a new model, e.g. `motionflow_mv/fusion/ray_attention_temporal_residual_occlusion_model.py`, subclassing `RayAttentionFusionModelTemporalResidual`.
2. After the temporal encoder (`ray_attention_temporal_residual_model.py:171`), add a small `occlusion_head` that outputs a logit `logit_o` of shape `(B*T, V, J)`.
3. Convert it to a soft visibility mask:

```python
visibility = 1 - torch.sigmoid(logit_o)   # (B*T, V, J)
```

4. Modify the DLT weight computation:

```python
weights = torch.sigmoid(w_logits).permute(0, 2, 1) * confidences * visibility
```

5. Add an auxiliary binary-cross-entropy loss against occlusion labels:
   - Generate labels from the existing dropout augmentation: when a detection is dropped, mark it occluded (`o=1`).
   - Use a small weight `λ_occ ≈ 0.1`.
6. Add a fallback guard: if a joint has fewer than two views with non-negligible visibility, fall back to unweighted DLT over all views or temporal interpolation.
7. Create a new training script by extending `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py` with the occlusion model and loss.
8. Evaluate with `experiments/eval_residual_robustness_mpiinf3dhp_v1.py` under occlusion rates `[0.0, 0.1, 0.3, 0.5, 0.7]`.

## 4. Expected success metric

- **Clean accuracy:** MPI-INF-3DHP cross-subject MPJPE stays ≤ 12 mm (baseline 11.17 mm).
- **Robustness:** At 30% random joint occlusion, MPJPE improves by ≥10% relative to the residual baseline.
- **Mask quality:** Occlusion prediction accuracy ≥ 80% on synthetic labels.
- **Runtime:** Overhead of the extra head < 5% on the existing RTX 4090 benchmark.

## 5. Risks / blockers

- **Synthetic vs. real occlusion mismatch:** Random dropout does not capture view-correlated, pose-dependent self-occlusion. Mitigate by using MPI-INF-3DHP confidence thresholds as proxy real labels.
- **Degenerate DLT:** Masking too many views can make the linear system ill-conditioned. The fallback guard is required.
- **Training instability:** The auxiliary BCE term may destabilize residual training. Start with `λ_occ = 0.1` and early stopping.
- **A800-D / Docker are read-only:** Do not modify anything there; run experiments on the local RTX 4090.
- **WebBridge data:** Download only if needed; do not commit large files.

---

**Intended report path:** `docs/swarm_iter7/occlusion_aware_fusion_mechanisms.md`