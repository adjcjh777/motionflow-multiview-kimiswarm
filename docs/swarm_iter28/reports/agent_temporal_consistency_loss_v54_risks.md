# v54 Temporal Consistency Loss — Risks and Mitigations

## Risk 1: Over-smoothing of fast actions

**Description** — A global velocity/acceleration penalty can suppress legitimate high-velocity motion (e.g. running, kicking), causing the model to trade per-frame accuracy for temporal smoothness.

**Mitigation** — Start with a very low weight (`v54_tc_loss_weight = 0.001`) and only ramp up after a fixed number of epochs (`v54_tc_warmup_epochs`). Use per-joint uncertainty from v52/v53 to mask out frames with low confidence, so the loss focuses on noisy rather than genuinely fast joints. Sweep the weight in `{0.001, 0.01, 0.05}` during smoke tests.

## Risk 2: Coupling with v52/v53 uncertainty estimates

**Description** — The temporal loss is meant to be weighted by per-joint uncertainty. If the v52/v53 uncertainty estimates are miscalibrated, the temporal loss will be mis-weighted and may reinforce errors.

**Mitigation** — Default `v54_tc_use_uncertainty: true` only after v52/v53 smoke metrics are stable. Provide a fallback path that uses uniform weights (`torch.ones`) so the module works even when v52/v53 are disabled. Log the mean effective weight per domain to catch calibration drift.

## Risk 3: Sequence-length dependence and edge cases

**Description** — The acceleration term requires `T >= 3`. Training clips can be as short as `T = 1` or contain missing frames, producing NaN gradients if not masked.

**Mitigation** — Compute velocity and acceleration only for valid consecutive frames using `torch.diff` with explicit length checks. When `T < 3`, set the acceleration term to `0` and the velocity term to the available frame pair. Normalize by the number of valid pairs, not by `T`.

## Risk 4: Redundancy with v47/v49 temporal modules

**Description** — v47 (temporal aggregation) and v49-Lite already add temporal modelling. Adding v54 as a loss may create double regularisation that over-constrains the pose trajectory.

**Mitigation** — Treat v54 as an optional add-on. Smoke tests should compare three configurations: (a) v53 alone, (b) v53 + v47, and (c) v53 + v54. If v47 is already enabled, keep `v54_tc_loss_weight` at the low end of the sweep. Document the interaction in `AGENTS.md` and do not enable v54 and v47 together in the first smoke.

## Risk 5: Identity-at-init is only approximate

**Description** — The module has no learnable weights, so it is nominally identity-at-init. However, because it adds a non-zero loss from the very first forward pass, loading a v53 checkpoint and enabling v54 will change gradients and optimisation dynamics.

**Mitigation** — Implement `v54_tc_warmup_epochs` so the loss is multiplied by `min(1.0, max(0.0, (epoch - warmup) / ramp_epochs))`. Set `ramp_epochs = 1` in smoke configs and `ramp_epochs = 2` for full runs. Verify that the first training step after loading a v53 checkpoint does not change `val_MPJPE` by more than `0.1` mm when `v54_tc_loss_weight = 0.0`.
