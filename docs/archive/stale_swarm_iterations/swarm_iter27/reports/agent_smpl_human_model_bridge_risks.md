# v53 SMPL Human-Model Bridge — Risk Register

**Owner:** design-swarm agent  
**Module:** `smpl_human_model_bridge_v53`  
**Date:** 2026-08-09

## R1: SMPL model file dependency or license friction

**Risk:** The module needs the SMPL body model (`SMPL_NEUTRAL.pkl`) or the `smplx` package.  If the file is missing or the license is not available on a headless A800 node, the module cannot be instantiated, blocking training and CI.

**Mitigation:**
- Make the SMPL forward optional.  When the model file is unavailable, fall back to a learned *parametric skeleton regressor* that predicts `P_smpl` directly and is trained with the same losses.
- Keep the fallback path default in smoke configs so RTX 4090 / GitHub Actions tests do not require the model file.
- Document the exact expected path (`data/smpl/SMPL_NEUTRAL.pkl`) and provide a one-time setup script for A800 nodes.

## R2: Skeleton mismatch and over-smoothing

**Risk:** SMPL has 24 joints while MotionFlow targets 17-joint H36M or 28-joint MPI-INF-3DHP skeletons.  A fixed regressor `M` can blur wrist/ankle detail or mis-map the spine, causing a net MPJPE regression.

**Mitigation:**
- Initialize `M` with the official SMPL-to-target joint regressor (e.g. H36M regressor) and freeze it for the first epoch.
- Clamp the blend gate `α` to a maximum value (`v53_smpl_max_gate = 0.8`) so the prior can never fully override strong fused evidence.
- Keep the residual MLP output zero-initialized and gated by `v53_smpl_residual_gate`, preserving identity at init.

## R3: Warm-start identity is not exact

**Risk:** Even with zero gates, the SMPL forward may produce a slightly different skeleton than the input `pred_3d_uwt` because the neutral SMPL shape is not the mean of the training set, breaking the warm-start guarantee.

**Mitigation:**
- In identity mode (`v53_smpl_identity_init=True`), bypass the SMPL forward during the first `v53_smpl_warmup_epochs` and set `P_smpl = P_uwt` exactly.
- Add a smoke test that loads a trained v52 checkpoint with v53 enabled and asserts `|val_MPJPE(v52) - val_MPJPE(v52+v53 at init)| < 0.1 mm`.
- Use a learned pose residual head that starts at zero rather than relying solely on the SMPL canonical pose.

## R4: Auxiliary losses conflict with v28/v40 physical-space alignment

**Risk:** The bone-length, floor-penetration, and self-intersection losses introduced by v53 may double-count or fight against the existing v28 physical-space alignment and v40 skeleton-aware physical loss, destabilizing training.

**Mitigation:**
- Start with very low loss weights (`v53_smpl_bone_weight=0.02`, `v53_smpl_floor_weight=0.01`, `v53_smpl_intersection_weight=0.005`) and only ramp them up after `v53_smpl_warmup_epochs`.
- Disable the v28/v40 bone-length/floor terms while tuning v53, then re-enable them one at a time.
- Log each auxiliary loss separately in the trainer so regressions can be isolated quickly.

## R5: Runtime and memory overhead from the SMPL forward

**Risk:** Running a full SMPL forward on every batch adds latency and memory, especially on the A800 with large `B` and `T`, and the 6890-vertex mesh is expensive to compute.

**Mitigation:**
- Cache the `smplx.SMPL` object per device and reuse it across iterations.
- Provide a fast mode (`v53_smpl_fast_mode=True`) that skips the 6890-vertex mesh forward and only predicts/regresses the `J` joints.
- Optionally apply the SMPL bridge only on a subsampled set of frames (e.g. every `v53_smpl_temporal_stride` frame) while blending per-frame.
- Profile a single forward pass before scaling to the full A800 batch size and adjust `batch_size` if needed.
