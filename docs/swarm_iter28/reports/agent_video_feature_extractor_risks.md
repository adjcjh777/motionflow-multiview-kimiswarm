# v54 Video Feature Extractor — Risk Register

## Risk 1: Factorized attention causes OOM on long clips or many views

- **Likelihood:** High
- **Impact:** The cross-view attention scales as \(O(B \cdot T \cdot J \cdot V^2 \cdot d_{\text{model}})\) and the skeleton graph mixer scales as \(O(B \cdot T \cdot V \cdot J^2 \cdot d_{\text{model}})\). With `clip_len=13`, `V=8`, and `J=17`, the attention maps can exceed the 24 GB RTX 4090 budget during smoke testing.
- **Mitigation:**
  - Default `v54_vfe_crossview_layers=1` and `v54_vfe_skeleton_layers=1`.
  - Use only 2 attention heads and `v54_vfe_d_model=64` by default.
  - Implement optional gradient checkpointing for the cross-view and skeleton branches behind a flag `v54_vfe_checkpoint_attn`.
  - Smoke with `clip_len=9` before raising to the full `clip_len` used in A800 runs.

## Risk 2: Physics gate collapses or ignores the physical signals

- **Likelihood:** Medium
- **Impact:** The per-joint physics MLP may learn a near-constant gate (e.g., all joints close to 0.5), so the module reduces to a single global update and wastes the physical-space signal. Alternatively, the gate may overfit to the training set and suppress updates on unseen poses.
- **Mitigation:**
  - Initialize the physics MLP final layer to zeros and the gate bias to 0.0; because the global gate `α` is also zero, identity-at-init is preserved.
  - Add a small L2 loss on the gate entropy to encourage it to span `[0.1, 0.9]`.
  - Log the mean and standard deviation of `γ_j` per joint in smoke/eval; if the gate standard deviation is below 0.05, halve the physics MLP hidden size and re-run smoke.

## Risk 3: Redundancy with v47/v49 temporal and v34/v36 graph modules

- **Likelihood:** Medium
- **Impact:** v54 refines features temporally and across the skeleton before triangulation, while v47/v49 refine poses temporally after triangulation and v34/v36 operate on per-frame view-joint graphs. Stacking all of them may cause over-parameterization, conflicting gradients, and no net MPJPE gain.
- **Mitigation:**
  - First smoke v54 on top of a bare v25/v52 baseline (no v47/v49/v34/v36) to isolate its contribution.
  - Run an ablation that disables v54 and re-enables v47/v49 to quantify overlap.
  - If redundancy is high, make the skeleton branch optional (`v54_vfe_use_skeleton_graph=false`) so v54 can act purely as a temporal/geometry feature extractor when graph modules are already enabled.

## Risk 4: Skeleton graph mixer breaks on mixed skeleton topologies

- **Likelihood:** Medium
- **Impact:** H36M uses 17 joints and MPI-INF-3DHP uses 28 joints. If a mixed batch reaches `OmniMultiViewFusionV5.forward`, a fixed `H36M_17_PARENTS` graph will produce shape mismatches or incorrect edges for the 28-joint sequences.
- **Mitigation:**
  - Infer the parent list from the number of joints at runtime; if `j == 17` use `H36M_17_PARENTS`, if `j == 28` use `MPI_INF_3DHP_28_PARENTS`, otherwise fall back to a simple kinematic chain.
  - Disable the skeleton branch automatically when `j` is unknown or mixed by setting `v54_vfe_use_skeleton_graph=false`.
  - Add a smoke test with a single 28-joint sample to catch topology issues early.

## Risk 5: Identity gate stays at zero and the module never trains

- **Likelihood:** Low–Medium
- **Impact:** The global gate `α` is initialized to 0.0 and all output projections are zero-initialized. If the optimizer or the warmup schedule keeps `α` near 0, v54 contributes nothing and the full A800 run wastes GPU time.
- **Mitigation:**
  - Start with `v54_vfe_warmup_epochs=0` so `α` is trainable from step 1, and initialize it to a small positive value (e.g., `1e-2`) only after identity-at-init smoke passes.
  - Log `α` every 100 steps; if it remains below `1e-3` after the first epoch, raise the initial value and rerun.
  - Add an auxiliary L2 penalty on `1 - α` with a tiny weight (e.g., `1e-4`) to encourage the gate to open once the main loss is stable.
