# v52 Learned View Selection Policy — Risk Register

## 1. Over-sparsity / budget collapse

**Risk:** The policy learns to drop so many views that triangulation becomes unstable, especially when the budget loss or temperature annealing pushes it toward very low `target_budget`.

**Mitigation:**
- Clamp the selection mask to `m ∈ [v52_lvsp_min_active, 1.0]` rather than allowing it to reach zero.
- Default `v52_lvsp_target_budget = 1.0` (no sparsity) and only enable the budget loss after `v52_lvsp_budget_warmup_epochs`.
- Enforce `v52_lvsp_min_views = 2` in the hard-selection path by masking out invalid subsets.

## 2. Gradient instability with hard view selection

**Risk:** If `v52_lvsp_hard=True` is used, straight-through Gumbel sampling can produce sparse or biased gradients, especially early in training when the model has not yet learned to rank views.

**Mitigation:**
- Keep the default as the continuous sigmoid mask (`v52_lvsp_hard=False`).
- If hard selection is needed for inference speed, anneal temperature from high to low and detach the hard mask before applying it to the DLT weights.

## 3. Redundancy with v46 and v51 reliability heads

**Risk:** v52 learns another per-view weighting on top of v46 sparse-view reliability and v51 cross-domain sparse-view reliability, leading to redundant gates and potential gradient conflicts.

**Mitigation:**
- Treat v52 as a *selection policy* rather than a reliability estimator: concatenate the v46/v51 reliability maps as inputs (`v52_lvsp_use_reliability_input=True`) instead of replacing them.
- Freeze v46/v51 weights during the first epoch while the policy warms up.
- Log the mean selection mask per view and per joint to detect whether v52 is learning the same ranking as v46/v51.

## 4. Camera-geometry overfitting

**Risk:** The policy may memorize dataset-specific camera layouts (e.g., Human3.6M vs. Shelf/Campus) instead of learning general view-quality cues.

**Mitigation:**
- Use a camera-conditioned embedding (`g_v`) rather than raw view indices, so the policy generalizes to unseen camera configurations.
- Apply mild random perturbations to `R`/`t` during training when `use_v52_learned_view_selection_policy=True`.
- Evaluate on a held-out camera setup in the smoke test before committing to the A800 queue.

## 5. Compute and memory overhead

**Risk:** Adding a cross-view transformer before triangulation increases peak GPU memory and latency, which can break the existing RTX 4090 smoke budget or A800 batch size.

**Mitigation:**
- Default to `v52_lvsp_hidden=64` and `v52_lvsp_n_layers=2` only; provide a lightweight MLP-only option `v52_lvsp_use_mlp_only=True` that omits the transformer for quick smoke tests.
- Project `feat` to a smaller `d_model` before the transformer if `C` is large.
- Benchmark wall-clock time in smoke; if overhead is >10 %, reduce `v52_lvsp_n_heads` or use a causal temporal window before full rollout.
