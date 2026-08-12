# v53 Cross-Dataset Domain Bridge — Risk Report

## Risk 1: Gate collapse / bridge overwrites the already-good v52 pose

**Concern:** The warm-start gate `α = sigmoid(-6.0) ≈ 0.0025` is small but not zero. With a large `v53_cdb_pose_loss_weight` the residual MLP can still learn a non-zero correction early and degrade the v52-UWT baseline, especially if the cross-domain attention misaligns domain prototypes.

**Mitigation:**
- Add a short warmup: `v53_cdb_warmup_epochs` starts the auxiliary loss at 0 and only enables it after the prototype bank has stabilized.
- Clamp the gate during the first epoch via `torch.clamp(α, max=0.1)` if the smoke shows baseline drift.
- Smoke-test with `v53_cdb_pose_loss_weight = 0.0` first to verify identity-like behavior; accept only if `val_MPJPE` is within 0.5 mm of v52-UWT.

## Risk 2: Domain-prototype bank overfits to source domains

**Concern:** The prototype bank `P ∈ R^{D×J×k}` is initialized from a source batch. If source and target domains have very different skeleton scales (e.g., H36M vs. 3DPW), the prototypes may overfit to source and hurt target-domain transfer.

**Mitigation:**
- Initialize `P` from a balanced mini-batch containing all available domains, not just the source.
- Add L2 regularization on `P` (weight `1e-4`) to prevent prototypes from drifting too far from the initial mean.
- Monitor per-domain validation MPJPE separately; if 3DPW degrades, fall back to the v52-UWT checkpoint by setting the gate `α = 0` during inference.

## Risk 3: Reprojection residual computed from `pred_3d_uwt` is noisy

**Concern:** The uncertainty-guided skeleton token uses the reprojection residual norm `r` as a geometry cue. If `pred_3d_uwt` is initially inaccurate, the residual is noisy and can mislead the domain bridge.

**Mitigation:**
- Stop-gradient on `r` so the bridge module does not chase moving reprojection targets through the v52-UWT output.
- Use the UWT weight `w_uwt` as a mask: zero-out the residual contribution for views where `w_uwt < v53_cdb_min_weight` (default `0.1`).
- Provide a fallback path: if `r` is unavailable (no camera params), the module still works by using only `x` and `log(w_uwt)`.

## Risk 4: GRL discriminator destabilizes training or fights v48

**Concern:** v53 introduces a second GRL domain discriminator after v48's discriminator. Multiple adversarial objectives can conflict, causing gradient interference, mode collapse, or NaN losses.

**Mitigation:**
- Make `v53_cdb_use_grl_discriminator` default `false`; enable only after the smoke shows stable training.
- Share the discriminator architecture with v48 if possible, or at least use a much smaller hidden dimension (`v53_cdb_hidden = 32`) to limit capacity.
- Scale the GRL loss by a small `v53_cdb_adv_loss_weight` (default `0.05`) and ramp it linearly after warmup.

## Risk 5: Additional module adds latency and memory at inference

**Concern:** v53 performs view-pooling, cross-domain attention, and an MLP at every forward pass. On A800 full runs with large clips, this may increase per-step time and memory enough to force a smaller batch size.

**Mitigation:**
- Keep the module small: one FiLM layer, one attention layer, and a single MLP layer with hidden dimension `64`.
- Cache the prototype bank and the FiLM parameters per domain so repeated forward passes for the same dataset reuse them.
- If profiling shows a >10% slowdown, optionally disable v53 during inference by loading the checkpoint and freezing the gate at `α = 0`, falling back to pure v52-UWT at test time.
