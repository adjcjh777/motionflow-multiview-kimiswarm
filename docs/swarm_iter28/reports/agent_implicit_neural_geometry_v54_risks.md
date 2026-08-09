# v54 Implicit Neural Geometry — Risk Report

**Module:** `ImplicitNeuralGeometryV54`  
**Location in pipeline:** After v53 Physical-Space Calibration in `OmniMultiViewFusionV5`  
**Related proposal:** `docs/swarm_iter28/proposals/v54_implicit_neural_geometry_v54.md`

---

## Risk 1: Identity-at-init is violated by the temporal or view-aggregation branch

**Description:** v54 must start as a no-op so that a v53 checkpoint can be loaded unchanged. If the final residual head, the ray-attention head, or the temporal MLP has any non-zero bias or if the residual gate does not start at zero, the first forward pass will shift the pose and break warm-start validation.

**Mitigation:**
- Initialize `MLP_α`, `MLP_s`, and `MLP_Δ` final layers with zeros.
- Use a residual gate `λ = sigmoid(g)` with `g` initialized to a large negative value (default `-6.0`, so `λ  0.002`).
- Add a unit test that loads a v53 checkpoint with v54 enabled and checks `val_MPJPE` changes by less than `0.1 mm`.
- Keep a smoke test variant with all v54 losses disabled to confirm the branch is structurally no-op.

---

## Risk 2: OOM or compute blow-up from the 4-D spatio-temporal field

**Description:** Building per-view, per-joint, per-temporal-frame embeddings and running MLPs over `(B, T, V, J, hidden)` tensors can exceed RTX 4090 / A800 memory, especially with `clip_len=243` or large `T`.

**Mitigation:**
- Default `temporal_window=3` (±1 frames) and use strided averaging when `T` is long.
- Run the spatial MLP once per `(B, T, V, J)` and share weights across time instead of materializing a 5-D intermediate for the temporal branch.
- Add an `max_temporal_len` guard that truncates or pools the temporal context to the same limit used by v47 (`max_temporal_len=256`).
- Profile memory in the smoke test; if peak GPU memory exceeds 80 %, reduce `hidden` or switch the temporal branch to causal Conv1D.

---

## Risk 3: Ray-alignment / reprojection loss destabilizes early training

**Description:** The ray and reprojection losses can pull the refined pose away from a good v53 initialization before the implicit field has learned a meaningful pose manifold, causing NaN/Inf or large MPJPE spikes in the first epoch.

**Mitigation:**
- Default `v54_ing_warmup_epochs=1` so the loss is active only after the v53 pose is stable.
- Start with low weights (`ray_loss_weight=0.1`, `reproj_loss_weight=1.0`) and a bounded residual (`Δp_j` via `tanh`).
- Clip gradients globally (`max_norm=1.0`) in the trainer when v54 is enabled.
- Add a smoke abort criterion: if `val_MPJPE` after epoch 1 is `> 2×` the v53 baseline, stop and reduce `reproj_loss_weight`.

---

## Risk 4: Surface-energy loss collapses to a trivial zero function

**Description:** The signed-distance energy `s_j` can trivially stay near zero for every pose, in which case `L_surface` no longer encodes a meaningful pose manifold and the module degenerates into a residual MLP.

**Mitigation:**
- Inject a small contrastive term during training: sample one negative pose per batch (randomly perturbed) and require its energy to be larger than the energy of the true pose by a margin.
- Monitor the variance of `s_j` across a validation batch; if variance drops below a threshold (`1e-4`), increase `surface_loss_weight` temporarily.
- Keep `use_surface_energy` toggleable; if ablation shows no gain, disable it without touching the rest of the architecture.

---

## Risk 5: Physical-constraint branches conflict with v53 PSC when both are active

**Description:** v54 can add its own bone-length and floor losses while v53 already enforces similar constraints. Running both may double-penalize the pose, causing stiff or over-constrained skeletons and worse MPJPE.

**Mitigation:**
- Default `v54_ing_use_physical_constraints=False` when `use_v53_physical_space_calibration=True`, then ablate turning it on.
- When both are on, scale v54's `bone_loss_weight` and `floor_loss_weight` down by `0.1×` relative to v53 and gate them with `v54_ing_warmup_epochs`.
- Use v54 physical losses only as soft residuals around the v53 estimate, never as primary constraints.
- Log the per-loss terms in TensorBoard; if v54 physical losses dominate, reduce their weights before the full A800 run.

---

## Summary checklist before A800 full run

- [ ] Identity-at-init test passes (`|MPJPE_v54 - MPJPE_v53| < 0.1 mm`).
- [ ] Smoke test completes without OOM on RTX 4090.
- [ ] No NaN/Inf in the first two epochs at smoke scale.
- [ ] `val_MPJPE@full` improves over v53 by at least `0.4 mm`.
- [ ] `MPJPE@k=2` and `MPJPE@k=3` are at least as good as v53.
- [ ] Ablations for `use_temporal_field`, `use_ray_alignment`, and `use_physical_constraints` are documented.
