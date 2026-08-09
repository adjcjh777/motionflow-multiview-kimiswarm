# Multi-View Video Data Augmentation Strategy for MotionFlow-MultiView

**Target:** ICRA/CVPR 2027 multi-view human pose estimation  
**Status:** Design proposal — no source files modified.  
**Scope:** Augmentations that operate on the canonical `(T, V, J, C)` 2-D keypoint/confidence tensor produced by `WebBridgeCanonical17Dataset` and `TemporalClipDataset`.

---

## 1. Current state

The project already contains a solid set of augmentation primitives:

| Module | What it does | Limitation for multi-view video |
|---|---|---|
| `motionflow_mv/data/sync_multiview_aug.py` | Horizontal flip, rotation, scaling, translation applied *identically* to all views | Per-frame only; no temporal or camera-geometry awareness |
| `motionflow_mv/data/synthetic_occlusion_aug.py` | Group- and joint-level occlusion, optional temporal consistency across frames | Operates on fixed real views only |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py::augment_clip` | Pixel noise, confidence dropout, random outlier injection, variable-view subset sampling | Generic; not informed by rig geometry or temporal motion |
| `docs/swarm_iter_next/v33_view_synthesis_augmentation.md` | Synthesizes virtual camera views from 3-D ground truth | A complete sub-project; needs a broader augmentation context |

The gap is a *unified multi-view video augmentation policy*: a set of transforms that (a) respect calibrated multi-view geometry, (b) exploit temporal continuity, and (c) create harder training examples without breaking epipolar or physical consistency.

---

## 2. Design principles

1. **Geometry-safe.** Any per-view transform must preserve epipolar consistency. The existing `SynchronizedMultiview2DAugmenter` is the right model: one random parameter set, applied identically to every view.
2. **Temporally coherent.** Video clips should be augmented as clips, not as independent frames. Occlusions, noise, and virtual views should persist across frames unless the goal is to simulate flickering sensors.
3. **Camera- and rig-aware.** Augmentation should know about baseline, field of view, and rig layout so it can generate realistic viewpoint variation rather than arbitrary 2-D warps.
4. **Task-aligned difficulty.** The hardest failure modes for multi-view pose are few-view fusion, calibration noise, and partial occlusion. Augmentations should target those failure modes.
5. **Composable and tunable.** Every transform is a standalone callable with CLI flags so ablations are one-line changes.

---

## 3. Proposed augmentation axes

### 3.1 Temporal clip-level geometric augmentation (`TemporalSynchronized2DAugmenter`)

Extend `SynchronizedMultiview2DAugmenter` so that the *same* geometric transform is applied to every frame of a clip, not just to one frame. This mimics a camera rig being physically rotated or zoomed, rather than random per-frame jitter.

```
Input:  x (T, V, J, C)
Sample one transform per clip:
  flip flag f ~ Bernoulli(p)
  angle ~ Uniform(-θ, +θ)
  scale ~ Uniform(s_min, s_max)
  translation ~ Uniform(-t, +t)^2
Apply identically to all T frames and all V views.
```

**Why:** Random per-frame rotation/scale teaches the model to tolerate noise, but the real-world analogue is rare. Clip-level transforms teach invariance to consistent viewpoint changes.

**Implementation sketch**

```python
# motionflow_mv/data/temporal_sync_multiview_aug.py
class TemporalSynchronized2DAugmenter:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, J, C) or (T, V, J, C)
        # sample once per batch/clip, apply to all T frames
        ...
```

**CLI flags (in training script)**

```text
--temporal_aug_prob 0.5          # probability of applying per-clip geom augment
--temporal_aug_rotation_deg 15.0
--temporal_aug_scale_range 0.9 1.1
--temporal_aug_translation_px 5.0
```

---

### 3.2 Camera geometry perturbation (`CameraPerturbationAugmenter`)

Small, realistic perturbations of `K, R, t` before feeding them into geometry-aware modules (e.g., v25 geometry fusion, v27 UDP, v33 ray attention). This makes the model robust to calibration drift and non-perfect extrinsics.

| Parameter | Perturbation | Typical range |
|---|---|---|
| Focal length `f` | `f * (1 + ε)` | `ε ~ N(0, 0.01)` |
| Principal point | `cx + δx, cy + δy` | `δ ~ N(0, 5 px)` |
| Rotation `R` | small axis-angle `ω ~ N(0, 0.5°)` | applied to each camera independently |
| Translation `t` | `t + Δt` | `Δt ~ N(0, 0.01 m)` |

**Constraints to keep training stable:**
- Perturb `K, R, t` consistently for an entire clip (temporal coherence).
- Do not perturb the *same* calibration used to generate 3-D supervision; perturb only the inputs to the model.
- Skip perturbation if `use_geometry_fusion` is disabled (no geometry module to consume it).

**Implementation sketch**

```python
# motionflow_mv/data/camera_perturbation_aug.py
def perturb_cameras(K, R, t, focal_sigma=0.01, pp_sigma_px=5.0,
                    rot_sigma_deg=0.5, t_sigma_m=0.01, temporal=True):
    ...
```

**CLI flags**

```text
--camera_perturb_prob 0.3
--camera_perturb_focal_sigma 0.01
--camera_perturb_pp_sigma_px 5.0
--camera_perturb_rot_sigma_deg 0.5
--camera_perturb_t_sigma_m 0.01
```

---

### 3.3 Structured multi-view occlusion (`StructuredMultiviewOcclusionAugmenter`)

The existing `SyntheticJointOcclusionAugmenter` already supports temporal consistency and group occlusion. Add multi-view structure:

1. **Same-joint multi-view occlusion:** a joint occluded in view `v` is more likely to be occluded in views with a similar viewing angle (e.g., nearby cameras). Simulates a body part hidden behind self-occlusion.
2. **Camera-specific streaks:** one or two entire camera views drop out for a contiguous sub-sequence of frames (e.g., a person walks behind a pillar for 10 frames).
3. **Reflection / ghost confidence:** for water/glass surfaces, a joint may have two weak detections. Augment by splitting confidence between true and a mirrored phantom location.

**Implementation sketch**

```python
# motionflow_mv/data/structured_multiview_occlusion_aug.py
class StructuredMultiviewOcclusionAugmenter:
    def __call__(self, x, view_mask=None, camera_positions=None):
        # x: (B, T, V, J, C)
        # 1. sample occlusion groups per clip
        # 2. correlate occlusions across views by camera angle
        # 3. optionally drop full views for T' contiguous frames
        return x_aug, view_mask_aug
```

**CLI flags**

```text
--structured_occlusion_prob 0.3
--structured_occlusion_group_rate 0.15
--structured_occlusion_view_streak_prob 0.05
--structured_occlusion_view_streak_max_frames 6
--structured_occlusion_correlation_radius_deg 45.0
```

---

### 3.4 Virtual-view synthesis as augmentation (leverage v33)

`v33_view_synthesis_augmentation.md` proposes rendering 2-D keypoints from virtual cameras. In the broader augmentation strategy, this becomes the *view-diversity* axis:

- Render `N_virtual` additional views from the same 3-D pose sequence.
- Mix them with real views in the view dimension.
- Re-use the same `view_mask` and `domain_mask` logic.

Recommended modes in order of complexity:

1. **Geometry mode:** pure pinhole projection with 2-D Gaussian noise and confidence decay based on reprojection angle.
2. **Skeleton-aware mode:** add bone-length and joint-limit priors to suppress anatomically implausible virtual views.
3. **Neural/hybrid mode:** learnable per-joint texture for more realistic 2-D appearance (future work).

This is the only augmentation axis that changes the number of input views, so it must be applied *after* variable-view subset sampling to avoid confusion.

---

### 3.5 Domain-mixing augmentation (`DomainMixAugmenter`)

The WebBridge mixed loader combines H36M, MPI-INF-3DHP, AIST++, Shelf, and Campus. Domain mixing at the batch level can improve cross-dataset generalization:

1. **Batch-level domain shuffling:** guarantee each batch contains at least two domains.
2. **Cross-domain pseudo-pairing:** pair a clip from one domain with calibration from another, then re-project. This is only valid if the canonical skeletons and scales match.
3. **Domain-specific noise calibration:** use domain-specific noise levels because MPI-INF-3DHP confidences are noisier than H36M.

---

## 4. Recommended augmentation schedule

Not all augmentations are active all the time. The following schedule is proposed:

| Phase | Epochs | Active augmentations | Purpose |
|---|---|---|---|
| Warm-up | 0–1 | Light pixel noise only | Let the model learn coarse geometry |
| Main training | 2+ | Pixel noise + temporal geom + structured occlusion + camera perturbation | Robust multi-view fusion |
| Late / specific | Selected epochs | Virtual-view synthesis + domain mixing | Viewpoint and domain generalization |

A curriculum parameter `augmentation_ramp_epochs` linearly increases the intensity of geometric and occlusion augmentations from 0 to the target values over the first `N` epochs.

---

## 5. Integration into the training pipeline

The proposed integration point is inside `experiments/train_omniview_fusion_v5_webbridge_multi.py`, immediately after the existing `augment_clip` call and before the model forward pass:

```python
# Existing per-clip augmentation
x, view_mask = augment_clip(x, ...)

# 1. Temporal synchronized geometric augmentation
if args.temporal_aug_prob > 0.0 and torch.rand(1) < args.temporal_aug_prob:
    x = temporal_sync_aug(x)

# 2. Camera perturbation (only for geometry-aware models)
if args.camera_perturb_prob > 0.0 and torch.rand(1) < args.camera_perturb_prob:
    K_aug, R_aug, t_aug = camera_perturb_aug(K, R, t)
else:
    K_aug, R_aug, t_aug = K, R, t

# 3. Structured multi-view occlusion
if args.structured_occlusion_prob > 0.0:
    x, view_mask = structured_occlusion_aug(x, view_mask, camera_positions=R, t=t)

# 4. Virtual-view synthesis (optional, from v33)
if args.use_view_synthesis_aug_v33:
    x, K_aug, R_aug, t_aug, view_mask = view_synthesis_aug(
        x, y, K_aug, R_aug, t_aug, view_mask, ...
    )

out = model(x, K=K_aug, R=R_aug, t=t_aug, view_mask=view_mask)
```

All augmentations are *in-place on a clone*, so the original loaded clip is preserved for the next epoch.

---

## 6. Evaluation protocol

To verify that the augmentation strategy actually improves multi-view pose estimation, the following diagnostics are proposed:

1. **Clean MPJPE:** standard benchmark on H36M / MPI-INF-3DHP test sets.
2. **Calibrated-perturbation MPJPE:** evaluate with `camera_perturb_*` noise injected at test time.
3. **Few-view robustness curve:** MPJPE as a function of `k` randomly sampled views (`k = 2, 4, 8, 14`).
4. **Occlusion robustness curve:** MPJPE as a function of increasing synthetic occlusion rate.
5. **Domain-transfer MPJPE:** train on H36M + MPI, evaluate on Shelf / Campus without fine-tuning.
6. **Reprojection consistency:** for virtual-view synthesis, measure 2-D reprojection error of predicted 3-D poses on held-out real camera views.

Minimum success criteria:
- No regression on clean MPJPE when all augmentations are disabled (`--temporal_aug_prob 0 --structured_occlusion_prob 0 --camera_perturb_prob 0`).
- ≥ 5% relative improvement on calibrated-perturbation and few-view metrics when augmentations are enabled.
- Monotonic few-view curve (`k=2` worst, `k=14` best) is preserved.

---

## 7. Implementation roadmap

| Step | Deliverable | Notes |
|---|---|---|
| 1 | `motionflow_mv/data/temporal_sync_multiview_aug.py` | Clip-level synchronized geometric augmentation |
| 2 | `motionflow_mv/data/camera_perturbation_aug.py` | Calibrated camera perturbation |
| 3 | `motionflow_mv/data/structured_multiview_occlusion_aug.py` | Multi-view correlated occlusion and view streaks |
| 4 | CLI flags in `train_omniview_fusion_v5_webbridge_multi.py` | Hook into existing training loop |
| 5 | Smoke tests on RTX 4090 | Verify no runtime errors and no NaNs |
| 6 | Ablation on H36M / MPI | Compare each axis independently and combined |
| 7 | Integrate v33 virtual-view synthesis | Adds view-diversity axis |

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Camera perturbation breaks triangulation/geometry fusion | Medium | Clamp perturbations; start with very small σ |
| Temporal geometric aug is too easy and adds no value | Low | Ablate against per-frame version |
| Structured occlusion is too harsh and slows convergence | Medium | Ramp in gradually; keep min-visible-joints guard |
| Virtual-view synthesis is redundant with variable-view training | Medium | Compare v33 + variable-view vs. each alone |
| Too many flags complicate ablations | Low | Provide a single `--augmentation_policy` preset |

---

## 9. Relation to existing and planned work

- Builds on `motionflow_mv/data/sync_multiview_aug.py` — adds temporal and camera-aware extensions.
- Complements `motionflow_mv/data/synthetic_occlusion_aug.py` — adds multi-view correlation rules.
- Uses the same canonical `(x, y, K, R, t)` format as `WebBridgeCanonical17Dataset` and `TemporalClipDataset`.
- Virtual-view synthesis is the v33 proposal; this document frames it as one axis of a larger multi-view video augmentation strategy.
- Camera perturbation directly supports the v25/v27/v28 geometry modules and the v31/v34 paper-story pipeline.
