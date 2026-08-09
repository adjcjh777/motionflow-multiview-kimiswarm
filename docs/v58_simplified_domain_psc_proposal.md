# v58 Simplified Domain-Conditional Physical-Space Calibration (SD-PSC) Proposal

## Motivation

v57 DC-PSC added per-domain FiLM conditioning to the floor, bone-scale, and residual heads, plus a per-domain residual gate. The local tiny smoke (train_samples=16, 2 epochs) regressed from v52's 78.68 mm to 102.38 mm val_MPJPE, likely because the extra ~42k domain-conditional parameters overfit the tiny dataset.

## Design

v58 keeps only the parts of domain conditioning that are physically meaningful and parameter-cheap:

1. **Per-domain canonical bone lengths**: each domain gets its own `canonical_log_bone_lengths` vector (same as v57/v53 already do). This is only `num_domains × n_bones` parameters.
2. **Per-domain floor height offset**: each domain gets a scalar floor offset added to the shared floor MLP output. Only `num_domains` parameters.
3. **Shared residual MLP and scalar gate**: the residual refinement head is shared across domains, with the same small scalar gate as v53. This removes the large per-domain FiLM layers and per-domain gate that likely caused overfitting.

This reduces the extra domain-specific parameter count from O(num_domains × hidden²) to O(num_domains × (n_bones + 1)).

## Module

`motionflow_mv/fusion/simplified_domain_psc_v58.py`

## Flags (to mirror v53/v57)

- `use_v58_simplified_domain_psc`
- `v58_sdpsc_hidden`, `v58_sdpsc_n_layers`
- `v58_sdpsc_num_domains`
- `v58_sdpsc_use_floor`, `v58_sdpsc_use_bone_scale`, `v58_sdpsc_use_uwt_weights`
- `v58_sdpsc_identity_init`, `v58_sdpsc_residual_gate_init`
- `v58_sdpsc_loss_weight`, `v58_sdpsc_warmup_epochs`
- `v58_sdpsc_floor_weight`, `v58_sdpsc_bone_weight`, `v58_sdpsc_reproj_weight`

## Validation plan

1. Identity-at-init: loading a v52 checkpoint with v58 enabled and loss_weight=0 should not change val_MPJPE by >0.1 mm.
2. Tiny smoke: run 2 epochs with train_samples=16. Accept if val_MPJPE is within 5 mm of the v52 baseline (78.68 mm).
3. Medium smoke: run 5 epochs with train_samples=200. Compare against v53 medium.
4. If promising, add to `launch_v33_a800_queue.py` and run full A800 training.\n## When to implement

If the v57 A800 full run (currently queued) does not beat the v53 A800 full run, implement v58 SD-PSC as the next iteration.
