# v52 Multi-View Temporal Sync — Risk Report

## Risk 1: Differentiable temporal warping degradates fast-motion features

**Description:** Linear interpolation of per-frame features along the time axis can blur high-frequency motion (quick limb movements). If the offset predictor predicts a fractional shift, the warped feature may lose sharp kinematic boundaries.

**Impact:** Wrist/ankle MPJPE could increase on fast-action clips.

**Mitigation:**
- Clamp `v52_mvts_max_shift` to a small value (default 3.0 frames).
- Add an L2 regularizer (`v52_mvts_offset_loss_weight`) to encourage small/zero offsets unless evidence is strong.
- Use nearest-neighbor warping as an ablation path; fallback to identity if the smoke shows regression.

---

## Risk 2: Cross-view temporal attention has quadratic memory in `V`

**Description:** The proposed same-time, same-joint cross-view attention computes a `V x V` attention matrix per `(b, t, j)` token. With `V = 8` and `J = 28`, this is still manageable, but combined with large batch/time dimensions it can OOM.

**Impact:** A800 smoke/full runs could hit GPU memory limits or slow down training.

**Mitigation:**
- Restrict attention to the same `(t, j)` slice only, never cross-time in this block.
- Use a default of `v52_mvts_n_heads=4` and keep `d_model=64`.
- Add an gradient-checkpointing option (`v52_mvts_checkpoint`) if memory becomes an issue.

---

## Risk 3: Identity-initialization fails to warm-start because of the gating mechanism

**Description:** The gated residual `out = (1 - g) * feat + g * feat_sync` is identity only if `g ≈ 0` at initialization. If the gate bias is wrong or the downstream loss pulls `g` away from zero too aggressively, the module can perturb an already-good v46/v49 baseline.

**Impact:** Smoke test could show a large MPJPE regression even though the module is "off" in theory.

**Mitigation:**
- Initialize gate bias so that `sigmoid(0) = 0.5` and then scale `g` by a small `eps` that ramps up via a warmup schedule.
- Alternatively, add a scalar `v52_mvts_residual_gain` initialized to `0.0` and linearly warmed up over the first epoch.
- Keep `v52_mvts_identity_init=true` as the default and only disable after verifying warm-start stability.

---

## Risk 4: Temporal offsets become dataset-specific instead of view-specific

**Description:** The offset predictor may learn to exploit domain-specific frame-rate or motion-style correlations rather than genuine camera synchronization. This limits cross-dataset generalization and conflicts with v48 domain-generalization goals.

**Impact:** Improved MPJPE on WebBridge but degraded or inconsistent results on H36M/MPI/3DPW.

**Mitigation:**
- Condition offset prediction on camera intrinsics/extrinsics (using the existing camera embedding path) rather than raw feature statistics, nudging the model toward geometry-based synchronization.
- Add a domain-discriminator adversarial loss (reuse v48 domain logits) to penalize dataset-specific offset distributions.
- Report per-domain offset histograms in smoke/eval to catch this early.

---

## Risk 5: Integration ordering with v50/v51 causes feedback loops

**Description:** v50 Self-Evolution Feedback Head and v51 CDSVR use per-view reliability/uncertainty computed from the final pose. Inserting v52 before the ST transformer changes the feature distribution entering v50/v51, potentially destabilizing their learned statistics.

**Impact:** Training instability, NaN/Inf in reliability heads, or inflated v50 auxiliary losses.

**Mitigation:**
- Run v52 smoke on top of the v46 baseline first, then with v50 enabled, then with v51 enabled, to isolate interactions.
- If instability appears, move the v52 block to run *after* v50/v51 (on the final pose) as a post-hoc temporal alignment instead of a feature-space module.
- Document the ordering constraint in `AGENTS.md` and the module docstring.
