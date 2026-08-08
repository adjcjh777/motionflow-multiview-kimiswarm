# Kinematic Anthropometric Prior (v22) — Concrete Improvements

**Scope:** `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py`  
**Context:** v23 = v18 + KAP (no neural BA). Goal: tighten KAP so it meaningfully lowers MPJPE without adding SMPL or heavy compute.

---

## 1. Make the bone-length prior adaptive per sample

**Status:** Implemented in `swarm/v22_kap_integration` — adds `adaptive_prior`, `adaptive_context_dim`, `adaptive_hidden`, and `adaptive_regularization` to `KinematicAnthropometricPrior`. The context encoder pools `feat_pooled` via mean+max; two zero-initialized MLPs predict per-bone adjustments to `bone_mu` and `bone_logvar`. An L2 regularization term keeps the module warm-startable.


**Observation:** `bone_mu` and `bone_logvar` are global population parameters (`kinematic_anthropometric_prior_v22.py:86-87`). A single prior cannot capture child/adult scale variation or camera-distance effects, so the residual branch has to do most of the scale correction.

**Proposal:** Condition the prior on the pooled per-joint features.

- Add a tiny context encoder that pools `feat_pooled` across joints (e.g. mean + max over J) into a latent `z ∈ R^c`.
- Predict per-bone adjustments:
  - `delta_mu = MLP_mu(z)`  → `mu_sample = bone_mu + delta_mu`
  - `delta_lv = MLP_lv(z)`  → `logvar_sample = bone_logvar + delta_lv`
- Keep `bone_mu` / `bone_logvar` as population anchors (regularize `delta_mu` and `delta_lv` toward zero with a small L2 penalty or use LayerNorm-style gating so the module stays warm-startable).

**Why:** A person-specific prior directly explains bone-scale variation, leaving the residual branch to correct fine pose errors rather than whole-body scale.

**Risk / mitigation:** Overfitting to subject scale in training data — keep `c ≤ 16` and add the regularization above; freeze the adjustment MLP for the first few epochs if loading a v18/v21 checkpoint.

**Files to touch:** `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py` (constructor `__init__`, `forward`).

---

## 2. Replace the fixed soft angle loss with a weighted + optionally projected angle constraint

**Observation:** The joint-limit term is added with an hard-coded unit weight (`kap_loss = bone_nll + angle_loss`, `kinematic_anthropometric_prior_v22.py:185-195`). There is no coefficient to balance the two objectives, and the penalty is purely soft, so implausible angles can still appear during inference if the network has not learned to respect them.

**Proposal:**

1. Add `angle_loss_weight: float = 0.1` to `__init__` and compute:
   ```python
   kap_loss = bone_nll + self.angle_loss_weight * angle_loss
   ```
2. Optionally implement a lightweight *hard* projection: after forming `pred_3d_refined`, identify any interior angle `θ > max_flexion_deg` and rotate the child bone direction until `θ = max_flexion_deg`. Make this a separate `project_angles=True/False` toggle so training stays differentiable while inference can be constrained.

**Why:** v21 (neural BA) already showed that unconstrained refinements can regress badly. Giving the angle limit its own, tunable weight and a hard safety net prevents KAP from producing anatomically impossible poses, especially for knees/elbows.

**Risk / mitigation:** Hard projection is non-differentiable; use it only at inference by default, and keep the soft penalty for training.

**Files to touch:** `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py`; expose `angle_loss_weight` and `kap_use_angle_limit` already wired in `omniview_fusion_v5.py:122-126`.

---

## 3. Add a bone-length preservation loss on the residual correction

**Observation:** The residual branch predicts a free 3-D delta per joint (`delta_head`) clamped globally by `tanh(...) * max_delta` (`kinematic_anthropometric_prior_v22.py:178-183`). Nothing stops the refinement from stretching or compressing bones relative to the input `pred_3d`, which conflicts with the very bone-length prior it is trying to satisfy.

**Proposal:** Compute an auxiliary *bone-length preservation* loss between the input pose and the refined pose:

```python
input_lengths = self._bone_lengths(pred_3d)
refined_lengths = self._bone_lengths(pred_3d_refined)
length_consistency = F.l1_loss(refined_lengths, input_lengths)  # or NLL
```

Add it to `kap_loss` with a small weight (e.g. `length_consistency_weight = 0.05`). This penalizes arbitrary skeleton stretching while still allowing the learned bone-length prior to pull the mean lengths toward plausible values.

**Why:** The prior pulls absolute lengths toward `bone_mu`, but the residual branch can move joints in ways that violate relative bone lengths. A consistency term couples the residual correction to the kinematic skeleton, making the refinement more stable.

**Risk / mitigation:** If the input `pred_3d` has wrong bone lengths, this term could fight the prior; keep its weight low (`≤ 0.05`) and only enable after the bone-length prior has had a few epochs to converge.

**Files to touch:** `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py` (`__init__`, `forward`); wire `kap_length_consistency_weight` through `omniview_fusion_v5.py` if adopted.

---

## Quick-win prioritization

1. **Highest expected impact:** Item 1 (adaptive prior) — directly addresses the biggest limitation of the current KAP.
2. **Safest guardrail:** Item 2 (weighted + projected angle limit) — minimal code, prevents regression.
3. **Refinement stability:** Item 3 (bone-length preservation) — complements item 1 by constraining how the residual branch uses its degrees of freedom.
