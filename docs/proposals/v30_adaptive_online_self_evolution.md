# v30: Adaptive Online Self-Evolution (AOSE) for Multi-View Pose

**Task identifier:** `design_v30_adaptive_online_self_evolution`  
**Depends on:** v29 (`docs/proposals/v29_self_evolving_hierarchical_multiview_fusion.md`)  
**Status:** Design / Candidate direction

## 1. Motivation

v29 adds a hierarchical view encoder, test-time self-evolution (TTE), and a physical-space training loss. Early smoke results are promising, but the model still treats every sample identically: it always runs the same number of TTE iterations and always uses the same hierarchical scale weights.

Drawing on the self-evolution ideas in the Qwen3 line of work, a stronger system should **adapt its own computation to the difficulty of each input** and **learn from its own predictions during training** without extra labels. v30 proposes two adaptive mechanisms:

1. **Adaptive TTE depth**: choose the number of self-evolution iterations based on per-sample residual uncertainty.
2. **Online self-distillation**: during training, generate pseudo-labels from the model's own multi-view predictions and use them to refine under-supervised joints.

## 2. Proposed method

### 2.1 Adaptive TTE depth (`AdaptiveTTEV30`)

Instead of a fixed `n_iters`, maintain a learned confidence threshold. For each sample, stop the TTE loop early when the mean reprojection residual drops below the threshold, and skip TTE entirely when the initial prediction is already confident.

Why it helps: easy samples get cheap inference; hard samples get more refinement. This is directly inspired by the "adaptive thinking" / "self-evolution" philosophy in Qwen3.

### 2.2 Online self-distillation (`OnlineSelfDistillationV30`)

During training, after the forward pass, compute a high-confidence pseudo-label by triangulating only the views with visibility > 0.8 and reprojection residual < 5 px. Add a small auxiliary loss:

```
L_self = |pred_3d - pseudo_label|_2 * confidence_mask
```

This loss is only active for joints with strong multi-view agreement, preventing drift.

Why it helps: it leverages the geometric self-consistency of multi-view data without requiring ground-truth 3D for every frame.

### 2.3 Dynamic scale selection in the hierarchical encoder

Replace the soft scale weights in v29 with a lightweight gating network that predicts per-sample scale weights from the input 2D keypoint statistics (spread, confidence, number of views).

## 3. Implementation plan

- `motionflow_mv/fusion/adaptive_online_self_evolution_v30.py`
  - `AdaptiveTTEV30`
  - `OnlineSelfDistillationV30`
  - `DynamicScaleGateV30`
- Wire flags in `OmniMultiViewFusionV5` and `train_omniview_fusion_v5_webbridge_multi.py`
- Smoke test + 1-epoch small run on 4090

## 4. Success criteria

- Smoke test passes.
- Local 4090 small-scale run shows val_MPJPE <= v29 or faster convergence.
- A800 full-scale run improves over v29 full.

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Adaptive TTE breaks gradient graph | Only used at inference; training uses fixed-depth TTE for stability |
| Self-distillation amplifies noise | Confidence mask + residual threshold |
| Dynamic gate overfits | Small MLP + dropout |
