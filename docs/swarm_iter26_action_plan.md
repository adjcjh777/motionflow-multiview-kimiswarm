# v56 Action Plan — Adaptive Physical Loss Weighting (APL) on v53

## Context

v53 Physical-Space Calibration (PSC) has shown promise:

- Local **tiny** smoke: 78.76 mm (same ballpark as v52 UWT 78.68 mm)
- Local **medium** run epoch-1: **48.24 mm** vs v52 UWT medium best **60.09 mm**
- v54 PSC-v2 and v55 ORR both failed tiny smoke (≈100 mm / 98 mm, worse than v52)

Decision: keep v53, drop v54/v55, and move the A800 full run forward.

## v56 Proposal

**Module:** `motionflow_mv/fusion/adaptive_physical_loss_v56.py`

**Idea:** instead of using fixed `v53_psc_floor_weight`, `v53_psc_bone_weight`, and `v53_psc_reproj_weight`, learn a small network that predicts per-sample (or per-joint) weights for the three PSC loss terms based on triangulation uncertainty and the current pose. The network is identity-at-init (outputs weights ≈ 1.0), so loading a v53 checkpoint into v56 does not change behavior.

**Why:** different domains and different body joints have different physical priors. A rigid 1.0 weight across all samples forces the same trade-off for easy and hard cases. Learning the weights should let v53 PSC focus on samples where the physical prior is reliable.

**Minimal architecture:**

```
Input: [uncertainty_mean, pose_std, domain_embedding, floor_loss_value, bone_loss_value, reproj_loss_value]
      -> 2-layer MLP with ReLU and residual gate
      -> outputs 3 logits (one per loss term)
      -> softplus + 1.0 => weights around 1.0 at init
```

The module is inserted in the trainer **after** the v53 PSC loss components are computed. The weighted loss is used for backpropagation. The auxiliary loss for monitoring remains the unweighted PSC loss.

## Flags

- `--use_v56_adaptive_physical_loss`
- `--v56_apl_hidden` (default 32)
- `--v56_apl_identity_init` (default True)
- `--v56_apl_loss_weight` (default 0.01)

## Workflow

1. Implement `AdaptivePhysicalLossV56`.
2. Wire into `experiments/train_omniview_fusion_v5_webbridge_multi.py`.
3. Create `scripts/run_v56_apl_smoke_local_4090.sh`.
4. Smoke test on RTX 4090 (tiny, 16 samples) on top of v53.
5. If tiny smoke is stable and within 1 mm of v53 tiny, add A800 queue entry and run medium/full.

## Risk

- Identity-at-init is critical; a bug here would make v56 worse than v53.
- The extra adaptive weights could overfit the tiny smoke. We must test on medium before committing to the A800 full run.
