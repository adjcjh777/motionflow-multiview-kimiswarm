# v52 Cross-Dataset Domain Bridge — Risk Register

## Risk 1: Identity gate is insufficient and the bridge degrades the already-good v45/v46/v51 baseline

**Likelihood:** Medium  
**Impact:** High

If the residual gate `α = sigmoid(-6.0)` is too small to learn quickly, the module remains near-identity and provides no benefit. If it learns too fast, the module can overwrite the strong v45/v46/v51 representations before the bridge loss has converged, causing a regression in `val_MPJPE`.

**Mitigation:**
- Use a **learnable, domain-specific gate schedule** rather than a single scalar: `α_d = sigmoid(g_d)` where `g_d` is learned per domain but initialized to `-6.0` for all source domains and `-4.0` for target domains.
- Add an **warmup epoch schedule**: set `v52_cdb_bridge_loss_weight = 0` for the first epoch, then ramp linearly to the target weight over the next two epochs.
- Smoke test against the v51-CDSVR baseline with identical seeds and reject the module if `val_MPJPE` is >2 mm worse.

## Risk 2: Prototype bank collapses or overfits to the source domain

**Likelihood:** Medium  
**Impact:** High

The learned domain-prototype bank `P ∈ R^{D×J×k}` can collapse if the contrastive loss pulls all joints to the same prototype, especially when target-domain samples are scarce (e.g., 3DPW). This removes the very domain discrimination the bridge is meant to provide.

**Mitigation:**
- Initialize prototypes from the **mean of a frozen v51 forward pass** over 512 source samples (`v52_cdb_use_prototype_init: true`), so the bank starts near the true data manifold.
- Add a **diversity regularizer** on the prototype bank: `L_div = -log det(P P^T + εI)` or, more cheaply, a pairwise distance penalty that enforces `||p_{d,j} - p_{d',j}||_2 > 0.1` for `d ≠ d'`.
- Monitor the **effective rank** of the prototype matrix during training; if it drops below `0.5·min(D·J, k)`, halve the learning rate of the prototype parameters.

## Risk 3: Contrastive bridge loss conflicts with the pose regression objective

**Likelihood:** Medium  
**Impact:** Medium

The bridge loss optimizes for domain-invariant representations, but the final MPJPE objective needs representations that preserve fine-grained 3D joint location information. Over-aggressive invariant learning can discard cues that are needed for accurate pose estimation.

**Mitigation:**
- Project the contrastive loss into a **separate, lower-dimensional bridge subspace** `z = W_b · x_bridge` with `W_b ∈ R^{d×d_b}` and `d_b = 32` (`v52_cdb_bridge_dim`), leaving the main `d`-dimensional feature stream mostly untouched.
- Share only the bridge subspace gradients back to `x_bridge`; keep the main head gradients isolated.
- Weight the bridge loss lightly (`v52_cdb_bridge_loss_weight = 0.005`) and increase only if cross-dataset validation improves.

## Risk 4: Geometry-preserving bone-length regularizer is too weak or too strong

**Likelihood:** Low–Medium  
**Impact:** Medium

The optional bone-length term can either fail to prevent the bridge from distorting skeleton geometry, or it can dominate early training and slow convergence of the domain-invariant features.

**Mitigation:**
- Make the bone-length loss **adaptive**: use the empirical bone-length mean `μ_bone` computed per batch from the ground-truth 3D pose, so the loss is scale-aware.
- Start with `v52_cdb_bone_loss_weight = 0.0` in smoke; enable only if the unconstrained bridge degrades physical-space alignment metrics (floor/bone error).
- Log the average bone-length error of the triangulated output and abort if it grows by more than 5% relative to baseline.

## Risk 5: Integration point in `OmniMultiViewFusionV5` creates a memory or runtime bottleneck

**Likelihood:** Low  
**Impact:** Medium

v52 introduces an extra cross-domain attention over `D×J×k` prototypes inside every forward pass. With `D=6`, `J=17`, `k=32`, the memory cost is negligible, but if the module is placed after temporal aggregation it operates on `(B,T,V,J,d)` tensors that can be large.

**Mitigation:**
- Place the module **after v51-CDSVR but before v47/v49 temporal aggregation**, so it operates on per-frame features (`T` is not larger than clip length).
- If memory is still a concern, run the cross-domain attention in a **time-shared loop** (process each frame independently) and cache the prototype bank across the clip.
- Benchmark GPU memory in the smoke run and fail the proposal if peak memory exceeds 110% of the v51-CDSVR baseline at the same batch size.
